"""Independent bounded exact-model oracle for the H6 weighted SMC recursion."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from vfe4.artifacts.atomic import (
    canonical_json_bytes as artifact_json_bytes,
    publish_run_directory,
)
from vfe4.artifacts.provenance import current_source_identity
from vfe4.data.windows import CausalPrefix
from vfe4.numerics.critical_values import (
    CRITICAL_VALUES_PROTOCOL_SHA256,
    FINITE_SMC_BIAS_LIMIT,
    FINITE_SMC_CELL_COUNT,
    FINITE_SMC_REPLICATE_COUNT,
    FINITE_SMC_SD_LIMIT,
    SmcErrorBounds,
    finite_smc_error_bounds,
)
from vfe4.predictive import (
    BootstrapSmcPredictor,
    CounterConsumption,
    CounterKey,
    CounterPurpose,
    EstimatorIdentity,
    EstimatorStream,
    ProposalPopulation,
    ProposalStep,
    vocabulary_identity_sha256,
)
from vfe4.types import EstimatorSpec, VocabularyIdentity
from vfe4.types.h6 import canonical_json_bytes


SMC_VALIDATION_RELATIVE_PATH = "validation/h6_smc_accuracy.json"
FINITE_FIXTURE_SHA256 = (
    "4d03f9b2f81743d816a17cfbc911ee9cdf24b24f34e61c03b050a8b7d6497117",
    "f6aa1faa93259518e6c634e0734d5e6a34ddc58f3d0e3024cc972d265cc06d0c",
    "920d4240d01bd24e6f650d119ce78b94dd0d58677d12e59e66cad6458ca34f89",
    "552a9a338e5ea2ace3964c0ff838038260c5103eabc63ca9d17e16875c9a2f9c",
)
_FULL_REPLICATE_SEEDS = tuple(range(2026072300, 2026072812))
_DEFERRED_OBLIGATION = (
    "deferred full gate requires four exact fixtures, 512 frozen seeds, "
    "six positions, and 256 particles"
)


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _probability_row(
    value: object, *, length: int, name: str
) -> tuple[float, ...]:
    if (
        type(value) is not list
        or len(value) != length
        or any(
            type(item) not in (int, float)
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in value
        )
    ):
        raise ValueError(f"{name} must be a finite probability row")
    row = tuple(float(item) for item in value)
    if not math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return row


@dataclass(frozen=True)
class FiniteSmcFixture:
    fixture_id: str
    raw_fixture_sha256: str
    vocabulary_id: str
    vocab_size: int
    horizon: int
    source_bank: Literal["state", "model"]
    source_variant: Literal["fixed", "prefix_conditioned"]
    initial_probabilities: tuple[float, ...]
    fixed_source_probabilities: tuple[float, ...] | None
    initial_source_probabilities: tuple[float, ...] | None
    prefix_source_probabilities: tuple[tuple[float, ...], ...] | None
    transition_kernels: tuple[
        tuple[tuple[float, ...], ...], ...
    ]
    emission_probabilities: tuple[tuple[float, ...], ...]
    observed_tokens: tuple[int, ...]
    semantic_sha256: str

    def __post_init__(self) -> None:
        if type(self.fixture_id) is not str or not self.fixture_id.startswith(
            "h6-smc-finite-"
        ):
            raise ValueError("finite fixture ID is not canonical")
        if (
            type(self.raw_fixture_sha256) is not str
            or len(self.raw_fixture_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.raw_fixture_sha256
            )
        ):
            raise ValueError("finite fixture raw hash must be SHA-256")
        if (
            type(self.vocab_size) is not int
            or self.vocab_size != 3
            or type(self.horizon) is not int
            or not 1 <= self.horizon <= 6
        ):
            raise ValueError("finite fixtures require V=3 and horizon in 1..6")
        if self.source_bank not in ("state", "model") or self.source_variant not in (
            "fixed",
            "prefix_conditioned",
        ):
            raise ValueError("finite source bank/variant is unsupported")
        state_count = len(self.initial_probabilities)
        if state_count < 2 or not math.isclose(
            math.fsum(self.initial_probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("finite initial law must be normalized")
        if (
            len(self.transition_kernels) != 2
            or any(len(kernel) != state_count for kernel in self.transition_kernels)
            or any(
                len(row) != state_count
                or not math.isclose(
                    math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-12
                )
                for kernel in self.transition_kernels
                for row in kernel
            )
        ):
            raise ValueError("finite transition kernels must be normalized square rows")
        if (
            len(self.emission_probabilities) != state_count
            or any(
                len(row) != self.vocab_size
                or not math.isclose(
                    math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-12
                )
                for row in self.emission_probabilities
            )
            or any(
                min(row[token] for row in self.emission_probabilities) <= 0.0
                for token in range(self.vocab_size)
            )
        ):
            raise ValueError("finite emission rows must be strictly positive and normalized")
        for token in range(self.vocab_size):
            column = tuple(row[token] for row in self.emission_probabilities)
            if max(column) / min(column) > 1.25 + 1e-15:
                raise ValueError("finite emission likelihood ratio exceeds 1.25")
        if (
            len(self.observed_tokens) != self.horizon
            or any(
                type(token) is not int or not 0 <= token < self.vocab_size
                for token in self.observed_tokens
            )
        ):
            raise ValueError("finite observed sequence does not match its horizon")
        if self.source_variant == "fixed":
            if (
                self.fixed_source_probabilities is None
                or self.initial_source_probabilities is not None
                or self.prefix_source_probabilities is not None
            ):
                raise ValueError("fixed source fixture has the wrong source fields")
        elif (
            self.fixed_source_probabilities is not None
            or self.initial_source_probabilities is None
            or self.prefix_source_probabilities is None
            or len(self.prefix_source_probabilities) != self.vocab_size
        ):
            raise ValueError("prefix source fixture has the wrong source fields")
        expected = _owned_hash(
            "vfe4.h6.finite-smc-fixture.v1", self._semantic_payload()
        )
        if self.semantic_sha256 != expected:
            raise ValueError("finite fixture semantic identity is stale")
        exact = exact_finite_oracle(self)
        if min(
            math.exp(value)
            for row in exact.token_log_probs
            for value in row
        ) < 0.10 - 1e-15:
            raise ValueError("finite exact token probability falls below 0.10")

    def _semantic_payload(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "raw_fixture_sha256": self.raw_fixture_sha256,
            "vocabulary_id": self.vocabulary_id,
            "vocab_size": self.vocab_size,
            "horizon": self.horizon,
            "source_bank": self.source_bank,
            "source_variant": self.source_variant,
            "initial_probabilities": self.initial_probabilities,
            "fixed_source_probabilities": self.fixed_source_probabilities,
            "initial_source_probabilities": self.initial_source_probabilities,
            "prefix_source_probabilities": self.prefix_source_probabilities,
            "transition_kernels": self.transition_kernels,
            "emission_probabilities": self.emission_probabilities,
            "observed_tokens": self.observed_tokens,
        }

    def source_probabilities(self, prefix_tokens: tuple[int, ...]) -> tuple[float, ...]:
        if self.source_variant == "fixed":
            assert self.fixed_source_probabilities is not None
            return self.fixed_source_probabilities
        if not prefix_tokens:
            assert self.initial_source_probabilities is not None
            return self.initial_source_probabilities
        assert self.prefix_source_probabilities is not None
        return self.prefix_source_probabilities[prefix_tokens[-1]]

    def truncate(self, horizon: int) -> "FiniteSmcFixture":
        if type(horizon) is not int or not 1 <= horizon <= self.horizon:
            raise ValueError("truncated horizon must lie inside the fixture")
        values = {
            **self._semantic_payload(),
            "horizon": horizon,
            "observed_tokens": self.observed_tokens[:horizon],
        }
        return FiniteSmcFixture(
            **values,
            semantic_sha256=_owned_hash(
                "vfe4.h6.finite-smc-fixture.v1", values
            ),
        )


def _load_finite_fixture_bytes(
    raw: bytes,
    *,
    filename: str,
) -> FiniteSmcFixture:
    if type(raw) is not bytes:
        raise ValueError("finite fixture snapshot must be immutable bytes")
    if type(filename) is not str or not filename:
        raise ValueError("finite fixture snapshot requires its filename")
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("finite fixture must be valid JSON") from exc
    if (
        type(payload) is not dict
        or payload.get("schema_version") != "h6-smc-finite-v1"
        or payload.get("enumeration") != "all_hidden_and_source_histories"
        or payload.get("vocab_size") != 3
        or payload.get("horizon") != 6
    ):
        raise ValueError("finite fixture schema is incomplete")
    fixture_id = payload.get("fixture_id")
    expected_name = f"{fixture_id.replace('-', '_')}.json" if type(fixture_id) is str else ""
    if filename != expected_name:
        raise ValueError("finite fixture filename does not match its ID")
    suffix = int(str(fixture_id).rsplit("-", 1)[-1])
    if not 1 <= suffix <= 4 or raw_sha256 != FINITE_FIXTURE_SHA256[suffix - 1]:
        raise ValueError("finite fixture raw hash is not the frozen hash")
    initial = _probability_row(
        payload.get("initial_probabilities"),
        length=len(payload.get("initial_probabilities", [])),
        name="initial_probabilities",
    )
    state_count = len(initial)
    emissions = tuple(
        _probability_row(row, length=3, name="emission row")
        for row in payload.get("emission_probabilities", [])
    )
    kernels = tuple(
        tuple(
            _probability_row(row, length=state_count, name="transition row")
            for row in kernel
        )
        for kernel in payload.get("transition_kernels", [])
    )
    variant = payload.get("source_variant")
    fixed = (
        _probability_row(
            payload.get("fixed_source_probabilities"),
            length=2,
            name="fixed_source_probabilities",
        )
        if variant == "fixed"
        else None
    )
    initial_source = (
        _probability_row(
            payload.get("initial_source_probabilities"),
            length=2,
            name="initial_source_probabilities",
        )
        if variant == "prefix_conditioned"
        else None
    )
    prefix_source = (
        tuple(
            _probability_row(row, length=2, name="prefix source row")
            for row in payload.get("prefix_source_probabilities", [])
        )
        if variant == "prefix_conditioned"
        else None
    )
    values = {
        "fixture_id": fixture_id,
        "raw_fixture_sha256": raw_sha256,
        "vocabulary_id": payload.get("vocabulary_id"),
        "vocab_size": 3,
        "horizon": 6,
        "source_bank": payload.get("source_bank"),
        "source_variant": variant,
        "initial_probabilities": initial,
        "fixed_source_probabilities": fixed,
        "initial_source_probabilities": initial_source,
        "prefix_source_probabilities": prefix_source,
        "transition_kernels": kernels,
        "emission_probabilities": emissions,
        "observed_tokens": tuple(payload.get("observed_tokens", [])),
    }
    return FiniteSmcFixture(
        **values,
        semantic_sha256=_owned_hash(
            "vfe4.h6.finite-smc-fixture.v1", values
        ),
    )


def _read_fixture_bytes(path: Path) -> bytes:
    return path.read_bytes()


def load_finite_fixture(path: Path) -> FiniteSmcFixture:
    if type(path) is not Path:
        path = Path(path)
    return _load_finite_fixture_bytes(
        _read_fixture_bytes(path),
        filename=path.name,
    )


@dataclass(frozen=True)
class ExactFiniteOracle:
    token_log_probs: tuple[tuple[float, ...], ...]
    sequence_log_normalizer: float


def exact_finite_oracle(fixture: FiniteSmcFixture) -> ExactFiniteOracle:
    """Enumerate the finite hidden/source mixture by exact filtering sums."""

    state_weights = fixture.initial_probabilities
    rows: list[tuple[float, ...]] = []
    log_normalizers: list[float] = []
    prefix: tuple[int, ...] = ()
    state_count = len(state_weights)
    for observed_token in fixture.observed_tokens:
        source_weights = fixture.source_probabilities(prefix)
        next_weights = tuple(
            math.fsum(
                state_weights[previous]
                * math.fsum(
                    source_weights[source]
                    * fixture.transition_kernels[source][previous][current]
                    for source in range(2)
                )
                for previous in range(state_count)
            )
            for current in range(state_count)
        )
        token_probabilities = tuple(
            math.fsum(
                next_weights[state]
                * fixture.emission_probabilities[state][token]
                for state in range(state_count)
            )
            for token in range(fixture.vocab_size)
        )
        rows.append(tuple(math.log(value) for value in token_probabilities))
        selected = token_probabilities[observed_token]
        log_normalizers.append(math.log(selected))
        state_weights = tuple(
            next_weights[state]
            * fixture.emission_probabilities[state][observed_token]
            / selected
            for state in range(state_count)
        )
        prefix += (observed_token,)
    return ExactFiniteOracle(
        tuple(rows), math.fsum(log_normalizers)
    )


class _FiniteProposalAdapter:
    proposal_mode = "generative_bootstrap"

    def __init__(self, fixture: FiniteSmcFixture) -> None:
        self.fixture = fixture
        self.vocabulary = VocabularyIdentity.from_tokenizer_spec(
            vocabulary_id=fixture.vocabulary_id,
            size=fixture.vocab_size,
            tokenizer_spec_bytes=b"VFE4-H6-FINITE-V3-TOKENIZER-V1\x00",
        )
        self.vocabulary_sha256 = vocabulary_identity_sha256(self.vocabulary)
        self.model_family_sha256 = _owned_hash(
            "vfe4.h6.finite-model-family.v1",
            {"fixture_semantic_sha256": fixture.semantic_sha256},
        )
        self.model_state_sha256 = _owned_hash(
            "vfe4.h6.finite-model-state.v1", fixture._semantic_payload()
        )
        self.proposal_identity_sha256 = _owned_hash(
            "vfe4.h6.finite-proposal.v1",
            {
                "proposal_mode": self.proposal_mode,
                "vocabulary_sha256": self.vocabulary_sha256,
                "model_family_sha256": self.model_family_sha256,
                "model_state_sha256": self.model_state_sha256,
            },
        )

    def assert_current_state(self) -> None:
        self.fixture.__post_init__()

    def initialize(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        particle_count: int,
    ) -> tuple[ProposalPopulation, tuple[CounterConsumption, ...]]:
        if (
            type(prefix_tokens) is not CausalPrefix
            or prefix_tokens.vocabulary != self.vocabulary
            or prefix_tokens.receiver_t != 1
        ):
            raise ValueError("finite initialization requires its empty CausalPrefix")
        log_initial = torch.log(
            torch.tensor(
                self.fixture.initial_probabilities, dtype=torch.float64
            )
        )
        states = [
            estimator_rng.categorical(
                CounterKey(
                    estimator_rng.stream_seed,
                    prefix_tokens.prefix_sha256,
                    0,
                    CounterPurpose.FINITE_INITIAL_CATEGORICAL,
                    particle,
                ),
                log_initial,
            )
            for particle in range(particle_count)
        ]
        population = ProposalPopulation.create(
            {
                "hidden_state_history": torch.tensor(
                    states, dtype=torch.int64
                ).unsqueeze(1)
            }
        )
        return population, (
            CounterConsumption.create(
                position=0,
                purpose=CounterPurpose.FINITE_INITIAL_CATEGORICAL,
                particle_count=particle_count,
                draws_per_particle=1,
            ),
        )

    def propagate(
        self,
        population: ProposalPopulation,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
    ) -> ProposalStep:
        position = prefix_tokens.receiver_t
        if (
            prefix_tokens.vocabulary != self.vocabulary
            or not 1 <= position <= self.fixture.horizon
        ):
            raise ValueError("finite propagation position is outside its fixture")
        history = population.component("hidden_state_history")
        if (
            history.dtype is not torch.int64
            or history.shape != (population.particle_count, position)
        ):
            raise ValueError("finite particle history does not match the prefix")
        source_probabilities = self.fixture.source_probabilities(
            tuple(int(value) for value in prefix_tokens.token_ids.tolist())
        )
        source_log_probs = torch.log(
            torch.tensor(source_probabilities, dtype=torch.float64)
        )
        source_purpose = (
            CounterPurpose.STATE_SOURCE_CATEGORICAL
            if self.fixture.source_bank == "state"
            else CounterPurpose.MODEL_SOURCE_CATEGORICAL
        )
        next_states: list[int] = []
        emissions: list[torch.Tensor] = []
        for particle in range(population.particle_count):
            source = estimator_rng.categorical(
                CounterKey(
                    estimator_rng.stream_seed,
                    prefix_tokens.prefix_sha256,
                    position,
                    source_purpose,
                    particle,
                ),
                source_log_probs,
            )
            previous = int(history[particle, -1].item())
            transition = torch.log(
                torch.tensor(
                    self.fixture.transition_kernels[source][previous],
                    dtype=torch.float64,
                )
            )
            current = estimator_rng.categorical(
                CounterKey(
                    estimator_rng.stream_seed,
                    prefix_tokens.prefix_sha256,
                    position,
                    CounterPurpose.FINITE_TRANSITION_CATEGORICAL,
                    particle,
                ),
                transition,
            )
            next_states.append(current)
            emissions.append(
                torch.log(
                    torch.tensor(
                        self.fixture.emission_probabilities[current],
                        dtype=torch.float64,
                    )
                )
            )
        next_history = torch.cat(
            [
                history,
                torch.tensor(next_states, dtype=torch.int64).unsqueeze(1),
            ],
            dim=1,
        )
        return ProposalStep.create(
            position=position,
            population=ProposalPopulation.create(
                {"hidden_state_history": next_history}
            ),
            emission_log_probs=torch.stack(emissions),
            counter_consumption=(
                CounterConsumption.create(
                    position=position,
                    purpose=source_purpose,
                    particle_count=population.particle_count,
                    draws_per_particle=1,
                ),
                CounterConsumption.create(
                    position=position,
                    purpose=CounterPurpose.FINITE_TRANSITION_CATEGORICAL,
                    particle_count=population.particle_count,
                    draws_per_particle=1,
                ),
            ),
            proposal_identity_sha256=self.proposal_identity_sha256,
        )


def build_finite_predictor(
    fixture: FiniteSmcFixture, *, particle_count: int
) -> tuple[BootstrapSmcPredictor, EstimatorIdentity]:
    fixture.__post_init__()
    spec = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=particle_count,
        resampling="systematic_ess_half",
    )
    identity = EstimatorIdentity.from_spec(spec)
    predictor = BootstrapSmcPredictor(
        proposal=_FiniteProposalAdapter(fixture),
        estimator_spec=spec,
        estimator_identity=identity,
        predictor_config_sha256=_owned_hash(
            "vfe4.h6.finite-predictor-config.v1",
            {"fixture_semantic_sha256": fixture.semantic_sha256},
        ),
        data_safety_sha256=fixture.raw_fixture_sha256,
    )
    return predictor, identity


def finite_gate_inventory() -> dict[str, int]:
    return {
        "fixture_count": 4,
        "horizon": 6,
        "vocabulary_size": 3,
        "token_cells": 4 * 6 * 3,
        "normalizer_cells": 4,
        "cell_count": FINITE_SMC_CELL_COUNT,
        "replicate_count": FINITE_SMC_REPLICATE_COUNT,
        "degrees_of_freedom": FINITE_SMC_REPLICATE_COUNT - 1,
        "tail_count": 2 * FINITE_SMC_CELL_COUNT * 2,
    }


def classify_smc_bounds(
    bounds: tuple[SmcErrorBounds, ...],
) -> Literal["PASS", "FAIL", "INCONCLUSIVE"]:
    if len(bounds) != FINITE_SMC_CELL_COUNT:
        raise ValueError("SMC classification requires the exact 76-cell inventory")
    if any(
        bound.lower_absolute_bias > FINITE_SMC_BIAS_LIMIT
        or bound.lower_sd > FINITE_SMC_SD_LIMIT
        for bound in bounds
    ):
        return "FAIL"
    if all(
        bound.upper_absolute_bias <= FINITE_SMC_BIAS_LIMIT
        and bound.upper_sd <= FINITE_SMC_SD_LIMIT
        for bound in bounds
    ):
        return "PASS"
    return "INCONCLUSIVE"


@dataclass(frozen=True)
class SmcAccuracyReport:
    gate: Literal["H6-SMC-Accuracy"]
    status: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    validation_path: Literal["validation/h6_smc_accuracy.json"]
    fixture_sha256: tuple[str, ...]
    estimator_semantic_sha256: str
    estimator_artifact_bytes_sha256: str
    critical_values_sha256: str
    executed_replicates: int
    executed_cells: int
    particle_count: int
    error_trace_sha256: str
    obligations: tuple[str, ...]
    report_sha256: str

    def __post_init__(self) -> None:
        if (
            self.gate != "H6-SMC-Accuracy"
            or self.validation_path != SMC_VALIDATION_RELATIVE_PATH
            or self.status not in ("PASS", "FAIL", "INCONCLUSIVE")
        ):
            raise ValueError("SMC report schema/status/path is invalid")
        if self.status == "PASS" and self.obligations:
            raise ValueError("PASS SMC report cannot retain obligations")
        if self.status == "INCONCLUSIVE" and not self.obligations:
            raise ValueError("INCONCLUSIVE SMC report requires an obligation")
        expected = _owned_hash(
            "vfe4.h6.smc-accuracy-report.v1", self._payload()
        )
        if self.report_sha256 != expected:
            raise ValueError("SMC accuracy report identity is stale")

    def _payload(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "status": self.status,
            "validation_path": self.validation_path,
            "fixture_sha256": self.fixture_sha256,
            "estimator_semantic_sha256": self.estimator_semantic_sha256,
            "estimator_artifact_bytes_sha256": (
                self.estimator_artifact_bytes_sha256
            ),
            "critical_values_sha256": self.critical_values_sha256,
            "executed_replicates": self.executed_replicates,
            "executed_cells": self.executed_cells,
            "particle_count": self.particle_count,
            "error_trace_sha256": self.error_trace_sha256,
            "obligations": self.obligations,
        }

    def artifact_bytes(self) -> bytes:
        self.__post_init__()
        return canonical_json_bytes(
            {**self._payload(), "report_sha256": self.report_sha256}
        )


def _validate_smc_grid_arguments(
    *,
    fixture_count: int,
    replicate_seeds: tuple[int, ...],
    particle_count: int,
) -> None:
    if (
        type(fixture_count) is not int
        or fixture_count <= 0
        or type(replicate_seeds) is not tuple
        or not replicate_seeds
        or any(
            type(seed) is not int or not 0 <= seed < 2**64
            for seed in replicate_seeds
        )
        or len(set(replicate_seeds)) != len(replicate_seeds)
        or type(particle_count) is not int
        or particle_count <= 0
    ):
        raise ValueError(
            "SMC gate requires explicit unique seeds, fixtures, and particles"
        )


def _snapshot_fixture_paths(
    fixture_paths: tuple[Path, ...],
) -> tuple[tuple[str, bytes], ...]:
    snapshots: list[tuple[str, bytes]] = []
    for value in fixture_paths:
        path = Path(value)
        snapshots.append((path.name, _read_fixture_bytes(path)))
    return tuple(snapshots)


def _run_h6_smc_gate_from_fixture_bytes(
    *,
    fixture_snapshots: tuple[tuple[str, bytes], ...],
    replicate_seeds: tuple[int, ...],
    particle_count: int,
    horizon_limit: int | None = None,
) -> SmcAccuracyReport:
    """Evaluate the existing SMC grid against one immutable fixture snapshot."""

    if (
        type(fixture_snapshots) is not tuple
        or any(
            type(snapshot) is not tuple
            or len(snapshot) != 2
            or type(snapshot[0]) is not str
            or type(snapshot[1]) is not bytes
            for snapshot in fixture_snapshots
        )
    ):
        raise ValueError("SMC fixture snapshots must be filename/bytes pairs")
    _validate_smc_grid_arguments(
        fixture_count=len(fixture_snapshots),
        replicate_seeds=replicate_seeds,
        particle_count=particle_count,
    )
    errors: dict[str, list[float]] = {}
    raw_hashes: list[str] = []
    identity: EstimatorIdentity | None = None
    for filename, raw_bytes in fixture_snapshots:
        fixture = _load_finite_fixture_bytes(
            raw_bytes,
            filename=filename,
        )
        raw_hashes.append(fixture.raw_fixture_sha256)
        if horizon_limit is not None:
            fixture = fixture.truncate(horizon_limit)
        exact = exact_finite_oracle(fixture)
        predictor, current_identity = build_finite_predictor(
            fixture, particle_count=particle_count
        )
        if identity is None:
            identity = current_identity
        elif identity != current_identity:
            raise ValueError("one SMC grid must use one estimator identity")
        for seed in replicate_seeds:
            stream = EstimatorStream.create(
                stream_seed=seed, estimator_identity=current_identity
            )
            prefix_tokens: list[int] = []
            cache = None
            selected_sum = 0.0
            for position, observed_token in enumerate(
                fixture.observed_tokens, start=1
            ):
                prefix = CausalPrefix.create(
                    receiver_t=position,
                    vocabulary=predictor.vocabulary,
                    token_ids=torch.tensor(
                        prefix_tokens, dtype=torch.int64
                    ),
                )
                prediction = predictor.next_token_log_probs(
                    prefix, stream, cache
                )
                values = prediction.log_probs.value()
                for token in range(fixture.vocab_size):
                    key = f"{fixture.fixture_id}:t{position}:v{token}"
                    errors.setdefault(key, []).append(
                        values[token].item()
                        - exact.token_log_probs[position - 1][token]
                    )
                selected_sum += values[observed_token].item()
                cache = prediction.cache
                prefix_tokens.append(observed_token)
            errors.setdefault(
                f"{fixture.fixture_id}:sequence_log_normalizer", []
            ).append(selected_sum - exact.sequence_log_normalizer)
    assert identity is not None
    full_inventory = (
        tuple(raw_hashes) == FINITE_FIXTURE_SHA256
        and replicate_seeds == _FULL_REPLICATE_SEEDS
        and particle_count == 256
        and horizon_limit in (None, 6)
        and len(errors) == FINITE_SMC_CELL_COUNT
    )
    if full_inventory:
        bounds = tuple(
            finite_smc_error_bounds(tuple(errors[key]))
            for key in sorted(errors)
        )
        status = classify_smc_bounds(bounds)
        obligations: tuple[str, ...] = (
            ()
            if status != "INCONCLUSIVE"
            else (
                "at least one simultaneous estimator bound crosses a frozen limit",
            )
        )
    else:
        status = "INCONCLUSIVE"
        obligations = (_DEFERRED_OBLIGATION,)
    error_trace_sha256 = _owned_hash(
        "vfe4.h6.smc-error-trace.v1",
        {
            key: tuple(values)
            for key, values in sorted(errors.items())
        },
    )
    payload = {
        "gate": "H6-SMC-Accuracy",
        "status": status,
        "validation_path": SMC_VALIDATION_RELATIVE_PATH,
        "fixture_sha256": tuple(raw_hashes),
        "estimator_semantic_sha256": identity.semantic_sha256,
        "estimator_artifact_bytes_sha256": identity.artifact_bytes_sha256,
        "critical_values_sha256": CRITICAL_VALUES_PROTOCOL_SHA256,
        "executed_replicates": len(replicate_seeds),
        "executed_cells": len(errors),
        "particle_count": particle_count,
        "error_trace_sha256": error_trace_sha256,
        "obligations": obligations,
    }
    return SmcAccuracyReport(
        **payload,
        report_sha256=_owned_hash(
            "vfe4.h6.smc-accuracy-report.v1", payload
        ),
    )


def run_h6_smc_gate(
    *,
    fixture_paths: tuple[Path, ...],
    replicate_seeds: tuple[int, ...],
    particle_count: int,
    horizon_limit: int | None = None,
    output_path: Path | None = None,
    repository_root: Path | None = None,
) -> SmcAccuracyReport:
    """Run an explicit grid; anything short of the frozen grid is non-closing."""

    if type(fixture_paths) is not tuple:
        raise ValueError(
            "SMC gate requires explicit unique seeds, fixtures, and particles"
        )
    _validate_smc_grid_arguments(
        fixture_count=len(fixture_paths),
        replicate_seeds=replicate_seeds,
        particle_count=particle_count,
    )
    destination: Path | None = None
    if output_path is not None:
        if repository_root is None:
            raise ValueError(
                "publishing an SMC artifact requires a declared repository root"
            )
        root = Path(repository_root).resolve()
        destination = Path(output_path).resolve()
        try:
            relative_destination = destination.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "SMC accuracy artifact must be inside the declared repository root"
            ) from error
        expected_parts = tuple(SMC_VALIDATION_RELATIVE_PATH.split("/"))
        if relative_destination.parts != expected_parts:
            raise ValueError(
                "SMC accuracy artifacts use validation/h6_smc_accuracy.json"
            )
    report = _run_h6_smc_gate_from_fixture_bytes(
        fixture_snapshots=_snapshot_fixture_paths(fixture_paths),
        replicate_seeds=replicate_seeds,
        particle_count=particle_count,
        horizon_limit=horizon_limit,
    )
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(report.artifact_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    return report


def _estimator_identity_for_particles(
    particle_count: int,
) -> EstimatorIdentity:
    return EstimatorIdentity.from_spec(
        EstimatorSpec.create(
            kind="weighted_smc",
            particle_count=particle_count,
            resampling="systematic_ess_half",
        )
    )


def _artifact_json_object(raw_bytes: bytes, *, name: str) -> dict[str, object]:
    if type(raw_bytes) is not bytes:
        raise ValueError(f"{name} must be immutable bytes")
    try:
        value = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if type(value) is not dict or artifact_json_bytes(value) != raw_bytes:
        raise ValueError(f"{name} must be a canonical JSON object")
    return value


def publish_h6_smc_accuracy_artifact(
    *,
    repository_root: Path,
    artifact_root: Path,
    run_name: str,
    fixture_paths: tuple[Path, ...],
    replicate_seeds: tuple[int, ...],
    particle_count: int,
    horizon_limit: int | None = None,
) -> tuple[SmcAccuracyReport, Path]:
    """Run from one snapshot and atomically publish direct readiness inputs."""

    if type(fixture_paths) is not tuple or len(fixture_paths) != 4:
        raise ValueError(
            "SMC readiness publication requires exactly four fixture paths"
        )
    _validate_smc_grid_arguments(
        fixture_count=len(fixture_paths),
        replicate_seeds=replicate_seeds,
        particle_count=particle_count,
    )
    repository = Path(repository_root).resolve()
    runs = Path(artifact_root)
    git_head_value, dirty_digest_value, source_sha256 = (
        current_source_identity(repository, runs)
    )

    estimator_identity = _estimator_identity_for_particles(particle_count)
    estimator_bytes = estimator_identity.artifact_bytes
    estimator_payload = _artifact_json_object(
        estimator_bytes,
        name="SMC estimator artifact",
    )
    fixture_snapshots = _snapshot_fixture_paths(fixture_paths)
    fixture_sha256 = tuple(
        hashlib.sha256(raw_bytes).hexdigest()
        for _, raw_bytes in fixture_snapshots
    )
    config_payload = {
        "schema_version": "h6-smc-accuracy-config-v1",
        "fixture_sha256": fixture_sha256,
        "replicate_seeds": replicate_seeds,
        "particle_count": particle_count,
        "horizon_limit": horizon_limit,
        "estimator_semantic_sha256": estimator_identity.semantic_sha256,
        "estimator_artifact_bytes_sha256": (
            estimator_identity.artifact_bytes_sha256
        ),
        "critical_values_sha256": CRITICAL_VALUES_PROTOCOL_SHA256,
    }
    fixture_set_payload = {
        "schema_version": "h6-finite-smc-fixture-set-v1",
        "encoding": "hex",
        "fixtures": tuple(
            {
                "filename": filename,
                "raw_sha256": raw_sha256,
                "raw_bytes_hex": raw_bytes.hex(),
            }
            for (filename, raw_bytes), raw_sha256 in zip(
                fixture_snapshots,
                fixture_sha256,
                strict=True,
            )
        ),
    }
    config_bytes = artifact_json_bytes(config_payload)
    fixture_set_bytes = artifact_json_bytes(fixture_set_payload)

    report = _run_h6_smc_gate_from_fixture_bytes(
        fixture_snapshots=fixture_snapshots,
        replicate_seeds=replicate_seeds,
        particle_count=particle_count,
        horizon_limit=horizon_limit,
    )
    if (
        report.fixture_sha256 != fixture_sha256
        or report.particle_count != particle_count
        or report.estimator_semantic_sha256
        != estimator_identity.semantic_sha256
        or report.estimator_artifact_bytes_sha256
        != estimator_identity.artifact_bytes_sha256
        or report.critical_values_sha256
        != CRITICAL_VALUES_PROTOCOL_SHA256
    ):
        raise ValueError(
            "SMC report does not match the snapshotted publication inputs"
        )

    validation_payload = {
        "schema_version": "vfe4-h6-smc-accuracy-v1",
        "gate": "H6-SMC-Accuracy",
        "git_head": git_head_value,
        "dirty_digest": dirty_digest_value,
        "source_sha256": source_sha256,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "estimator_sha256": estimator_identity.artifact_bytes_sha256,
        "estimator_semantic_sha256": estimator_identity.semantic_sha256,
        "fixture_set_sha256": hashlib.sha256(
            fixture_set_bytes
        ).hexdigest(),
        "status": report.status.lower(),
        "obligations": report.obligations,
        "producer_validation": _artifact_json_object(
            report.artifact_bytes(),
            name="SMC accuracy report",
        ),
    }
    run_directory = publish_run_directory(
        runs,
        run_name,
        {
            "config.json": config_payload,
            "protocol/estimator.json": estimator_payload,
            "fixtures/finite_smc.json": fixture_set_payload,
            SMC_VALIDATION_RELATIVE_PATH: validation_payload,
        },
    )
    return report, run_directory


__all__ = [
    "FINITE_FIXTURE_SHA256",
    "SMC_VALIDATION_RELATIVE_PATH",
    "ExactFiniteOracle",
    "FiniteSmcFixture",
    "SmcAccuracyReport",
    "build_finite_predictor",
    "classify_smc_bounds",
    "exact_finite_oracle",
    "finite_gate_inventory",
    "load_finite_fixture",
    "publish_h6_smc_accuracy_artifact",
    "run_h6_smc_gate",
]
