"""Source-bound continuous bootstrap-SMC sensitivity gate for H6.

The production entry point consumes the pinned T=32, d=16 decimal fixture but
is never run at import time.  The small entry point exists only for the
T=3, d=2, N=(8,16), two-stream unit contract.  Neither report can transfer a
bias sign or numerical bound to a trained language-model endpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import verification.numpy_oracles.h6_linear_gaussian_smc as _oracle_module
from verification.numpy_oracles.h6_linear_gaussian_smc import (
    PINNED_T32_D16_RELATIVE_PATH,
    PINNED_T32_D16_SHA256,
    LinearGaussianFixture,
    LinearGaussianSmcCalibrationReport,
    LinearGaussianSmcRawRecord,
    build_continuous_sensitivity_report,
    kalman_log_likelihood,
    load_pinned_linear_gaussian_fixture,
)


ALGORITHM_IDENTITY = (
    "bootstrap-carried-weight-systematic-ess-half-common-philox-v1"
)
RNG_IDENTITY = "numpy.random.Philox"
DTYPE_IDENTITY = "<f8"
RESAMPLING_IDENTITY = "systematic_when_ess_le_half"
Q2_REMAINDER_CONTRACTION = 0.75
PRODUCTION_PARTICLE_COUNTS = (128, 256, 512, 1024)
PRODUCTION_STREAM_SEEDS = tuple(
    202607250000 + index for index in range(64)
)
TINY_PARTICLE_COUNTS = (8, 16)
TINY_STREAM_SEEDS = (2026072501, 2026072502)
TRANSFER_CLAIM = "sensitivity_control_not_trained_endpoint_bound"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _owned_hash(domain: bytes, payload: object) -> str:
    return hashlib.sha256(domain + b"\x00" + _canonical_bytes(payload)).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ContinuousSmcRunConfig:
    schema_version: Literal["h6-continuous-smc-config-v1"]
    profile: Literal["unit_t3_d2", "production_t32_d16"]
    fixture_sha256: str
    particle_counts: tuple[int, ...]
    stream_seeds: tuple[int, ...]
    algorithm_identity: Literal[
        "bootstrap-carried-weight-systematic-ess-half-common-philox-v1"
    ]
    rng_identity: Literal["numpy.random.Philox"]
    dtype_identity: Literal["<f8"]
    resampling_identity: Literal["systematic_when_ess_le_half"]
    ess_threshold_fraction: Literal[0.5]
    common_random_block_particle_count: int
    gate_source_sha256: str
    oracle_source_sha256: str
    config_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "h6-continuous-smc-config-v1":
            raise ValueError("continuous SMC config schema is frozen")
        _require_sha256(self.fixture_sha256, "fixture_sha256")
        _require_sha256(self.gate_source_sha256, "gate_source_sha256")
        _require_sha256(self.oracle_source_sha256, "oracle_source_sha256")
        if (
            self.algorithm_identity != ALGORITHM_IDENTITY
            or self.rng_identity != RNG_IDENTITY
            or self.dtype_identity != DTYPE_IDENTITY
            or self.resampling_identity != RESAMPLING_IDENTITY
            or self.ess_threshold_fraction != 0.5
        ):
            raise ValueError("continuous SMC algorithm contract is stale")
        expected_inventory = {
            "unit_t3_d2": (
                TINY_PARTICLE_COUNTS,
                TINY_STREAM_SEEDS,
                None,
            ),
            "production_t32_d16": (
                PRODUCTION_PARTICLE_COUNTS,
                PRODUCTION_STREAM_SEEDS,
                PINNED_T32_D16_SHA256,
            ),
        }.get(self.profile)
        if expected_inventory is None:
            raise ValueError("continuous SMC profile is unknown")
        counts, seeds, pinned_sha256 = expected_inventory
        if (
            self.particle_counts != counts
            or self.stream_seeds != seeds
            or self.common_random_block_particle_count != max(counts)
            or (
                pinned_sha256 is not None
                and self.fixture_sha256 != pinned_sha256
            )
        ):
            raise ValueError("continuous SMC profile inventory is stale")
        if self.config_sha256 != _config_sha256(self):
            raise ValueError("continuous SMC config hash is stale")


def _config_payload(config: ContinuousSmcRunConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "profile": config.profile,
        "fixture_sha256": config.fixture_sha256,
        "particle_counts": config.particle_counts,
        "stream_seeds": config.stream_seeds,
        "algorithm_identity": config.algorithm_identity,
        "rng_identity": config.rng_identity,
        "dtype_identity": config.dtype_identity,
        "resampling_identity": config.resampling_identity,
        "ess_threshold_fraction": config.ess_threshold_fraction.hex(),
        "common_random_block_particle_count": (
            config.common_random_block_particle_count
        ),
        "gate_source_sha256": config.gate_source_sha256,
        "oracle_source_sha256": config.oracle_source_sha256,
    }


def _config_sha256(config: ContinuousSmcRunConfig) -> str:
    return _owned_hash(
        b"vfe4.h6.continuous-smc-config.v1",
        _config_payload(config),
    )


def build_continuous_smc_config(
    fixture: LinearGaussianFixture,
    *,
    profile: Literal["unit_t3_d2", "production_t32_d16"],
) -> ContinuousSmcRunConfig:
    """Bind a frozen profile to its fixture and current source bytes."""

    if type(fixture) is not LinearGaussianFixture:
        raise ValueError("continuous SMC config requires exact fixture")
    fixture.__post_init__()
    expected_shape = {
        "unit_t3_d2": (3, 2, 2),
        "production_t32_d16": (32, 16, 16),
    }.get(profile)
    if expected_shape is None or (
        fixture.time_steps,
        fixture.state_dimension,
        fixture.observation_dimension,
    ) != expected_shape:
        raise ValueError("fixture shape does not match continuous SMC profile")
    particle_counts, stream_seeds = {
        "unit_t3_d2": (TINY_PARTICLE_COUNTS, TINY_STREAM_SEEDS),
        "production_t32_d16": (
            PRODUCTION_PARTICLE_COUNTS,
            PRODUCTION_STREAM_SEEDS,
        ),
    }[profile]
    values: dict[str, object] = {
        "schema_version": "h6-continuous-smc-config-v1",
        "profile": profile,
        "fixture_sha256": fixture.fixture_sha256,
        "particle_counts": particle_counts,
        "stream_seeds": stream_seeds,
        "algorithm_identity": ALGORITHM_IDENTITY,
        "rng_identity": RNG_IDENTITY,
        "dtype_identity": DTYPE_IDENTITY,
        "resampling_identity": RESAMPLING_IDENTITY,
        "ess_threshold_fraction": 0.5,
        "common_random_block_particle_count": max(particle_counts),
        "gate_source_sha256": _source_sha256(Path(__file__)),
        "oracle_source_sha256": _source_sha256(
            Path(_oracle_module.__file__)
        ),
    }
    provisional = object.__new__(ContinuousSmcRunConfig)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return ContinuousSmcRunConfig(
        **values,
        config_sha256=_config_sha256(provisional),
    )


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    result = maximum + math.log(float(np.exp(values - maximum).sum()))
    if not math.isfinite(result):
        raise ValueError("particle log-sum-exp is nonfinite")
    return result


def _observation_log_likelihoods(
    particles: np.ndarray,
    observation: np.ndarray,
    observation_matrix: np.ndarray,
    observation_cholesky: np.ndarray,
) -> np.ndarray:
    residuals = observation[None, :] - particles @ observation_matrix.T
    diagonal = np.diag(observation_cholesky)
    if not np.array_equal(
        observation_cholesky,
        np.diag(diagonal),
    ):
        raise ValueError(
            "frozen continuous SMC fixtures require diagonal observation noise"
        )
    whitened = residuals / diagonal[None, :]
    dimension = observation.shape[0]
    constant = dimension * math.log(2.0 * math.pi) + 2.0 * float(
        np.log(np.diag(observation_cholesky)).sum()
    )
    return -0.5 * (constant + np.square(whitened).sum(axis=1))


def _systematic_indices(
    weights: np.ndarray,
    uniform: float,
) -> np.ndarray:
    particle_count = weights.shape[0]
    positions = (
        uniform / particle_count
        + np.arange(particle_count, dtype=np.float64) / particle_count
    )
    cumulative = np.cumsum(weights, dtype=np.float64)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _bootstrap_log_likelihood(
    fixture: LinearGaussianFixture,
    *,
    particle_count: int,
    seed: int,
    common_random_block_particle_count: int,
) -> float:
    """Run one carried-weight bootstrap filter under a fixed random schedule."""

    generator = np.random.Generator(np.random.Philox(seed))
    initial_mean = np.asarray(fixture.initial_mean, dtype=np.float64)
    initial_cholesky = np.linalg.cholesky(
        np.asarray(fixture.initial_covariance, dtype=np.float64)
    )
    transition = np.asarray(fixture.transition_matrix, dtype=np.float64)
    process_cholesky = np.linalg.cholesky(
        np.asarray(fixture.process_covariance, dtype=np.float64)
    )
    observation_matrix = np.asarray(
        fixture.observation_matrix,
        dtype=np.float64,
    )
    observation_cholesky = np.linalg.cholesky(
        np.asarray(fixture.observation_covariance, dtype=np.float64)
    )
    observations = np.asarray(fixture.observations, dtype=np.float64)
    initial_noise = generator.standard_normal(
        (common_random_block_particle_count, fixture.state_dimension)
    )[:particle_count]
    particles = initial_mean[None, :] + initial_noise @ initial_cholesky.T
    log_weights = np.full(
        particle_count,
        -math.log(particle_count),
        dtype=np.float64,
    )
    log_normalizer = 0.0

    for time_index, observation in enumerate(observations):
        if time_index:
            process_noise = generator.standard_normal(
                (
                    common_random_block_particle_count,
                    fixture.state_dimension,
                )
            )[:particle_count]
            particles = (
                particles @ transition.T
                + process_noise @ process_cholesky.T
            )
        candidate = log_weights + _observation_log_likelihoods(
            particles,
            observation,
            observation_matrix,
            observation_cholesky,
        )
        increment = _logsumexp(candidate)
        log_normalizer += increment
        log_weights = candidate - increment
        weights = np.exp(log_weights)
        ess = 1.0 / float(np.square(weights).sum())
        systematic_uniform = float(generator.random())
        if (
            time_index + 1 < fixture.time_steps
            and ess <= 0.5 * particle_count
        ):
            particles = particles[
                _systematic_indices(weights, systematic_uniform)
            ]
            log_weights.fill(-math.log(particle_count))

    if not math.isfinite(log_normalizer):
        raise ValueError("continuous particle log likelihood is nonfinite")
    return float(log_normalizer)


def _raw_by_key(
    config: ContinuousSmcRunConfig,
    raw_records: tuple[LinearGaussianSmcRawRecord, ...],
) -> dict[tuple[int, int], float]:
    if type(raw_records) is not tuple:
        raise ValueError("continuous raw records must be a tuple")
    for record in raw_records:
        if type(record) is not LinearGaussianSmcRawRecord:
            raise ValueError("continuous raw records must be exact records")
        record.__post_init__()
    expected_order = tuple(
        (stream_id, particle_count)
        for stream_id in range(len(config.stream_seeds))
        for particle_count in config.particle_counts
    )
    observed_order = tuple(
        (record.stream_id, record.particle_count)
        for record in raw_records
    )
    if observed_order != expected_order:
        raise ValueError("continuous raw ladder inventory is stale")
    return {
        (record.stream_id, record.particle_count): (
            record.raw_log_likelihood
        )
        for record in raw_records
    }


def _derive_ladder(
    config: ContinuousSmcRunConfig,
    raw_records: tuple[LinearGaussianSmcRawRecord, ...],
) -> tuple[
    tuple[tuple[int, float], ...],
    tuple[tuple[float, ...], ...],
    bool,
    float | None,
    float | None,
    float | None,
    bool | None,
    float,
]:
    by_key = _raw_by_key(config, raw_records)
    raw_means = tuple(
        (
            particle_count,
            math.fsum(
                by_key[(stream_id, particle_count)]
                for stream_id in range(len(config.stream_seeds))
            )
            / len(config.stream_seeds),
        )
        for particle_count in config.particle_counts
    )
    q_levels = tuple(
        tuple(
            2.0 * by_key[(stream_id, upper)]
            - by_key[(stream_id, lower)]
            for stream_id in range(len(config.stream_seeds))
        )
        for lower, upper in zip(
            config.particle_counts[:-1],
            config.particle_counts[1:],
            strict=True,
        )
    )
    q2 = q_levels[-1]
    q2_mean = math.fsum(q2) / len(q2)
    stream_radius = max(abs(value - q2_mean) for value in q2)
    if len(q_levels) < 3:
        return (
            raw_means,
            q_levels,
            False,
            None,
            None,
            None,
            None,
            stream_radius,
        )
    r1 = tuple(
        right - left
        for left, right in zip(q_levels[-3], q_levels[-2], strict=True)
    )
    r2 = tuple(
        right - left
        for left, right in zip(q_levels[-2], q_levels[-1], strict=True)
    )
    u1 = max(abs(value) for value in r1)
    u2 = max(abs(value) for value in r2)
    bias_bound = u2 / (1.0 - Q2_REMAINDER_CONTRACTION)
    contraction_eligible = u2 <= Q2_REMAINDER_CONTRACTION * u1
    return (
        raw_means,
        q_levels,
        True,
        u1,
        u2,
        bias_bound,
        contraction_eligible,
        stream_radius + bias_bound,
    )


@dataclass(frozen=True, slots=True)
class ContinuousSmcGateReport:
    schema_version: Literal["h6-continuous-smc-gate-v1"]
    config: ContinuousSmcRunConfig
    exact_log_likelihood: float
    raw_records: tuple[LinearGaussianSmcRawRecord, ...]
    raw_means: tuple[tuple[int, float], ...]
    q_levels: tuple[tuple[float, ...], ...]
    q2_sensitivity: LinearGaussianSmcCalibrationReport
    contraction_evaluated: bool
    contraction_u1: float | None
    contraction_u2: float | None
    conditional_bias_bound: float | None
    contraction_eligible: bool | None
    q2_two_sided_radius: float
    q2_interval_lower: float
    q2_interval_upper: float
    q2_interval_contains_exact: bool
    q2_bound_kind: Literal["two_sided_sensitivity_envelope"]
    transfer_claim: Literal[
        "sensitivity_control_not_trained_endpoint_bound"
    ]
    status: Literal["PASS", "INCONCLUSIVE"]
    obligations: tuple[str, ...]
    report_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "h6-continuous-smc-gate-v1":
            raise ValueError("continuous SMC gate schema is frozen")
        if type(self.config) is not ContinuousSmcRunConfig:
            raise ValueError("continuous SMC gate requires exact config")
        self.config.__post_init__()
        if (
            type(self.exact_log_likelihood) is not float
            or not math.isfinite(self.exact_log_likelihood)
        ):
            raise ValueError("exact continuous likelihood must be finite")
        derived = _derive_ladder(self.config, self.raw_records)
        (
            raw_means,
            q_levels,
            contraction_evaluated,
            u1,
            u2,
            bias_bound,
            contraction_eligible,
            radius,
        ) = derived
        q2 = q_levels[-1]
        q2_mean = math.fsum(q2) / len(q2)
        lower = q2_mean - radius
        upper = q2_mean + radius
        contains = lower <= self.exact_log_likelihood <= upper
        top_counts = self.config.particle_counts[-2:]
        top_raw = tuple(
            record
            for record in self.raw_records
            if record.particle_count in top_counts
        )
        self.q2_sensitivity.__post_init__()
        expected_status = (
            "PASS"
            if contraction_evaluated
            and contraction_eligible is True
            and contains
            else "INCONCLUSIVE"
        )
        expected_obligations: list[str] = []
        if not contraction_evaluated:
            expected_obligations.append(
                "two-level unit profile does not evaluate contraction"
            )
        elif contraction_eligible is not True:
            expected_obligations.append(
                "continuous Q2 remainder contraction did not close"
            )
        if not contains:
            expected_obligations.append(
                "two-sided continuous sensitivity envelope missed exact value"
            )
        if (
            self.raw_means != raw_means
            or self.q_levels != q_levels
            or self.contraction_evaluated is not contraction_evaluated
            or self.contraction_u1 != u1
            or self.contraction_u2 != u2
            or self.conditional_bias_bound != bias_bound
            or self.contraction_eligible is not contraction_eligible
            or self.q2_two_sided_radius != radius
            or self.q2_interval_lower != lower
            or self.q2_interval_upper != upper
            or self.q2_interval_contains_exact is not contains
            or self.q2_sensitivity.raw_records != top_raw
            or self.q2_sensitivity.fixture_sha256
            != self.config.fixture_sha256
            or self.q2_sensitivity.exact_log_likelihood
            != self.exact_log_likelihood
            or self.q2_sensitivity.q2_mean != q2_mean
            or self.q2_sensitivity.q2_two_sided_radius != radius
            or self.q2_sensitivity.q2_interval_lower != lower
            or self.q2_sensitivity.q2_interval_upper != upper
            or self.q2_sensitivity.q2_interval_contains_exact
            is not contains
            or self.q2_bound_kind != "two_sided_sensitivity_envelope"
            or self.transfer_claim != TRANSFER_CLAIM
            or self.status != expected_status
            or self.obligations != tuple(expected_obligations)
        ):
            raise ValueError("continuous SMC gate fields are inconsistent")
        if self.report_sha256 != _gate_report_sha256(self):
            raise ValueError("continuous SMC gate report hash is stale")


def _gate_report_payload(
    report: ContinuousSmcGateReport,
) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "config_sha256": report.config.config_sha256,
        "exact_log_likelihood": report.exact_log_likelihood.hex(),
        "raw_records": tuple(
            (
                record.stream_id,
                record.particle_count,
                record.raw_log_likelihood.hex(),
            )
            for record in report.raw_records
        ),
        "raw_means": tuple(
            (particle_count, value.hex())
            for particle_count, value in report.raw_means
        ),
        "q_levels": tuple(
            tuple(value.hex() for value in level)
            for level in report.q_levels
        ),
        "q2_sensitivity_sha256": report.q2_sensitivity.report_sha256,
        "contraction_evaluated": report.contraction_evaluated,
        "contraction_u1": (
            None
            if report.contraction_u1 is None
            else report.contraction_u1.hex()
        ),
        "contraction_u2": (
            None
            if report.contraction_u2 is None
            else report.contraction_u2.hex()
        ),
        "conditional_bias_bound": (
            None
            if report.conditional_bias_bound is None
            else report.conditional_bias_bound.hex()
        ),
        "contraction_eligible": report.contraction_eligible,
        "q2_two_sided_radius": report.q2_two_sided_radius.hex(),
        "q2_interval_lower": report.q2_interval_lower.hex(),
        "q2_interval_upper": report.q2_interval_upper.hex(),
        "q2_interval_contains_exact": report.q2_interval_contains_exact,
        "q2_bound_kind": report.q2_bound_kind,
        "transfer_claim": report.transfer_claim,
        "status": report.status,
        "obligations": report.obligations,
    }


def _gate_report_sha256(report: ContinuousSmcGateReport) -> str:
    return _owned_hash(
        b"vfe4.h6.continuous-smc-gate.v1",
        _gate_report_payload(report),
    )


def run_continuous_smc_gate(
    fixture: LinearGaussianFixture,
    config: ContinuousSmcRunConfig,
) -> ContinuousSmcGateReport:
    """Execute one explicitly constructed unit or production inventory."""

    if type(fixture) is not LinearGaussianFixture:
        raise ValueError("continuous SMC gate requires exact fixture")
    fixture.__post_init__()
    if type(config) is not ContinuousSmcRunConfig:
        raise ValueError("continuous SMC gate requires exact config")
    config.__post_init__()
    if fixture.fixture_sha256 != config.fixture_sha256:
        raise ValueError("continuous SMC fixture/config identity mismatch")
    if config.gate_source_sha256 != _source_sha256(Path(__file__)):
        raise ValueError("continuous SMC gate source changed after config")
    if config.oracle_source_sha256 != _source_sha256(
        Path(_oracle_module.__file__)
    ):
        raise ValueError("continuous SMC oracle source changed after config")

    raw_records = tuple(
        LinearGaussianSmcRawRecord(
            stream_id=stream_id,
            particle_count=particle_count,
            raw_log_likelihood=_bootstrap_log_likelihood(
                fixture,
                particle_count=particle_count,
                seed=seed,
                common_random_block_particle_count=(
                    config.common_random_block_particle_count
                ),
            ),
        )
        for stream_id, seed in enumerate(config.stream_seeds)
        for particle_count in config.particle_counts
    )
    (
        raw_means,
        q_levels,
        contraction_evaluated,
        u1,
        u2,
        bias_bound,
        contraction_eligible,
        radius,
    ) = _derive_ladder(config, raw_records)
    exact = kalman_log_likelihood(fixture)
    top_counts = config.particle_counts[-2:]
    top_raw = tuple(
        record
        for record in raw_records
        if record.particle_count in top_counts
    )
    q2_sensitivity = build_continuous_sensitivity_report(
        fixture=fixture,
        raw_records=top_raw,
        lower_particle_count=top_counts[0],
        upper_particle_count=top_counts[1],
        q2_two_sided_radius=radius,
    )
    lower = q2_sensitivity.q2_interval_lower
    upper = q2_sensitivity.q2_interval_upper
    contains = lower <= exact <= upper
    status: Literal["PASS", "INCONCLUSIVE"] = (
        "PASS"
        if contraction_evaluated
        and contraction_eligible is True
        and contains
        else "INCONCLUSIVE"
    )
    obligations: list[str] = []
    if not contraction_evaluated:
        obligations.append(
            "two-level unit profile does not evaluate contraction"
        )
    elif contraction_eligible is not True:
        obligations.append(
            "continuous Q2 remainder contraction did not close"
        )
    if not contains:
        obligations.append(
            "two-sided continuous sensitivity envelope missed exact value"
        )
    values: dict[str, object] = {
        "schema_version": "h6-continuous-smc-gate-v1",
        "config": config,
        "exact_log_likelihood": exact,
        "raw_records": raw_records,
        "raw_means": raw_means,
        "q_levels": q_levels,
        "q2_sensitivity": q2_sensitivity,
        "contraction_evaluated": contraction_evaluated,
        "contraction_u1": u1,
        "contraction_u2": u2,
        "conditional_bias_bound": bias_bound,
        "contraction_eligible": contraction_eligible,
        "q2_two_sided_radius": radius,
        "q2_interval_lower": lower,
        "q2_interval_upper": upper,
        "q2_interval_contains_exact": contains,
        "q2_bound_kind": "two_sided_sensitivity_envelope",
        "transfer_claim": TRANSFER_CLAIM,
        "status": status,
        "obligations": tuple(obligations),
    }
    provisional = object.__new__(ContinuousSmcGateReport)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    return ContinuousSmcGateReport(
        **values,
        report_sha256=_gate_report_sha256(provisional),
    )


def run_tiny_continuous_smc_gate(
    fixture: LinearGaussianFixture,
) -> ContinuousSmcGateReport:
    """Run only the frozen T=3, d=2, N=(8,16), two-stream unit profile."""

    return run_continuous_smc_gate(
        fixture,
        build_continuous_smc_config(fixture, profile="unit_t3_d2"),
    )


def run_pinned_production_continuous_smc_gate(
    repository_root: Path,
) -> ContinuousSmcGateReport:
    """Explicitly run the pinned 64-stream production sensitivity inventory."""

    if not isinstance(repository_root, Path):
        raise ValueError("repository_root must be pathlib.Path")
    fixture = load_pinned_linear_gaussian_fixture(
        repository_root / PINNED_T32_D16_RELATIVE_PATH
    )
    config = build_continuous_smc_config(
        fixture,
        profile="production_t32_d16",
    )
    return run_continuous_smc_gate(fixture, config)


__all__ = [
    "ALGORITHM_IDENTITY",
    "ContinuousSmcGateReport",
    "ContinuousSmcRunConfig",
    "DTYPE_IDENTITY",
    "PRODUCTION_PARTICLE_COUNTS",
    "PRODUCTION_STREAM_SEEDS",
    "RESAMPLING_IDENTITY",
    "RNG_IDENTITY",
    "TINY_PARTICLE_COUNTS",
    "TINY_STREAM_SEEDS",
    "TRANSFER_CLAIM",
    "build_continuous_smc_config",
    "run_continuous_smc_gate",
    "run_pinned_production_continuous_smc_gate",
    "run_tiny_continuous_smc_gate",
]
