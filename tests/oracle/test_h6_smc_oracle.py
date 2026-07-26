from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from vfe4.data.windows import CausalPrefix
from vfe4.predictive import (
    CounterKey,
    CounterPurpose,
    EstimatorStream,
    assimilate_log_weights,
    systematic_ancestors,
    weighted_mixture_log_probs,
)
from verification.h6_smc_gate import (
    FINITE_FIXTURE_SHA256,
    build_finite_predictor,
    exact_finite_oracle,
    load_finite_fixture,
    run_h6_smc_gate,
)


FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "verification" / "fixtures"
)
EXPECTED_FIXTURE_SHA256 = (
    "4d03f9b2f81743d816a17cfbc911ee9cdf24b24f34e61c03b050a8b7d6497117",
    "f6aa1faa93259518e6c634e0734d5e6a34ddc58f3d0e3024cc972d265cc06d0c",
    "920d4240d01bd24e6f650d119ce78b94dd0d58677d12e59e66cad6458ca34f89",
    "552a9a338e5ea2ace3964c0ff838038260c5103eabc63ca9d17e16875c9a2f9c",
)


def test_finite_fixture_checkout_bytes_match_frozen_inventory() -> None:
    fixture_paths = tuple(
        FIXTURE_ROOT / f"h6_smc_finite_{index:02d}.json"
        for index in range(1, 5)
    )
    assert tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in fixture_paths
    ) == FINITE_FIXTURE_SHA256


def test_weighted_recursion_and_counter_rules_match_independent_numpy() -> None:
    log_weights = torch.log(
        torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)
    )
    emissions = torch.log(
        torch.tensor(
            [
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
                [0.1, 0.2, 0.7],
            ],
            dtype=torch.float64,
        )
    )
    production = weighted_mixture_log_probs(log_weights, emissions)
    expected = np.log(
        np.array([0.7, 0.2, 0.1])
        @ np.array(
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]
        )
    )
    assert production.tolist() == pytest.approx(expected.tolist(), abs=1e-15)
    assert not torch.equal(
        production,
        torch.logsumexp(emissions, dim=0) - math.log(3),
    )

    update = assimilate_log_weights(log_weights, emissions[:, 1])
    unnormalized = np.array([0.7, 0.2, 0.1]) * np.array([0.3, 0.5, 0.2])
    expected_log_z = math.log(float(unnormalized.sum()))
    expected_weights = unnormalized / unnormalized.sum()
    assert update.log_normalizer == pytest.approx(expected_log_z, abs=1e-15)
    assert update.normalized_log_weights.tolist() == pytest.approx(
        np.log(expected_weights).tolist(), abs=1e-15
    )
    assert update.ess == pytest.approx(
        1.0 / float(np.square(expected_weights).sum()), abs=1e-14
    )

    ancestors = systematic_ancestors(
        torch.log(torch.tensor([0.6, 0.25, 0.15], dtype=torch.float64)),
        offset=0.04,
    )
    independent = np.searchsorted(
        np.array([0.6, 0.85, 1.0]),
        0.04 + np.arange(3) / 3.0,
        side="left",
    )
    assert ancestors.tolist() == independent.tolist()

    fixture_digest = "1" * 64
    fixture = load_finite_fixture(
        FIXTURE_ROOT / "h6_smc_finite_01.json"
    ).truncate(2)
    _, identity = build_finite_predictor(fixture, particle_count=8)
    stream = EstimatorStream.create(stream_seed=9, estimator_identity=identity)
    key = CounterKey(
        stream_seed=9,
        prefix_sha256=fixture_digest,
        position=1,
        purpose=CounterPurpose.FINITE_TRANSITION_CATEGORICAL,
        particle_index=0,
    )
    uniforms = tuple(stream.open_uniform(key, draw_index=i) for i in range(9))
    assert all(0.0 < value < 1.0 for value in uniforms)
    assert uniforms == tuple(
        stream.open_uniform(key, draw_index=i) for i in range(9)
    )
    gaussian_key = CounterKey(
        stream_seed=9,
        prefix_sha256=fixture_digest,
        position=1,
        purpose=CounterPurpose.STATE_TRANSITION_GAUSSIAN,
        particle_index=0,
    )
    assert stream.gaussian(gaussian_key, count=5) == stream.gaussian(
        gaussian_key, count=5
    )
    assert stream.categorical(
        key,
        torch.log(torch.tensor([0.0, 0.25, 0.75], dtype=torch.float64)),
    ) in (1, 2)
    with pytest.raises(ValueError, match="purpose"):
        CounterKey(9, fixture_digest, 1, "ad_hoc", 0)  # type: ignore[arg-type]


def test_small_finite_oracle_cold_warm_replay_and_deferred_inventory() -> None:
    fixture_paths = tuple(
        FIXTURE_ROOT / f"h6_smc_finite_{index:02d}.json"
        for index in range(1, 5)
    )
    assert tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in fixture_paths
    ) == EXPECTED_FIXTURE_SHA256
    assert FINITE_FIXTURE_SHA256 == EXPECTED_FIXTURE_SHA256

    fixture = load_finite_fixture(fixture_paths[1]).truncate(3)
    exact = exact_finite_oracle(fixture)
    predictor, identity = build_finite_predictor(fixture, particle_count=24)
    replicate_rows: list[list[list[float]]] = []
    replicate_log_z: list[float] = []

    for seed in (101, 102):
        stream = EstimatorStream.create(
            stream_seed=seed, estimator_identity=identity
        )
        tokens: list[int] = []
        cache = None
        rows: list[list[float]] = []
        selected = 0.0
        for token in fixture.observed_tokens:
            prefix = CausalPrefix.create(
                receiver_t=len(tokens) + 1,
                vocabulary=predictor.vocabulary,
                token_ids=torch.tensor(tokens, dtype=torch.int64),
            )
            prediction = predictor.next_token_log_probs(prefix, stream, cache)
            rows.append(prediction.log_probs.value().tolist())
            selected += prediction.log_probs.value()[token].item()
            cache = prediction.cache
            tokens.append(token)
        replicate_rows.append(rows)
        replicate_log_z.append(selected)

    mean_rows = np.mean(np.asarray(replicate_rows), axis=0)
    np.testing.assert_allclose(
        mean_rows,
        np.asarray(exact.token_log_probs),
        rtol=0.0,
        atol=0.04,
    )
    assert float(np.mean(replicate_log_z)) == pytest.approx(
        exact.sequence_log_normalizer, abs=0.08
    )

    stream = EstimatorStream.create(
        stream_seed=101, estimator_identity=identity
    )
    empty = CausalPrefix.create(
        receiver_t=1,
        vocabulary=predictor.vocabulary,
        token_ids=torch.empty(0, dtype=torch.int64),
    )
    first = predictor.next_token_log_probs(empty, stream)
    one_token = CausalPrefix.create(
        receiver_t=2,
        vocabulary=predictor.vocabulary,
        token_ids=torch.tensor([fixture.observed_tokens[0]], dtype=torch.int64),
    )
    warm = predictor.next_token_log_probs(one_token, stream, first.cache)
    cold = predictor.next_token_log_probs(
        one_token,
        EstimatorStream.create(
            stream_seed=101, estimator_identity=identity
        ),
    )
    assert warm.log_probs.raw_bytes_sha256 == cold.log_probs.raw_bytes_sha256
    assert warm.cache.cache_sha256 == cold.cache.cache_sha256

    report = run_h6_smc_gate(
        fixture_paths=(fixture_paths[0],),
        replicate_seeds=(11, 12),
        particle_count=8,
        horizon_limit=2,
    )
    assert report.status == "INCONCLUSIVE"
    assert report.executed_replicates == 2
    assert report.obligations == (
        "deferred full gate requires four exact fixtures, 512 frozen seeds, "
        "six positions, and 256 particles",
    )


def test_smc_artifact_path_is_exactly_under_declared_repository_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="validation/h6_smc_accuracy.json"):
        run_h6_smc_gate(
            fixture_paths=(FIXTURE_ROOT / "h6_smc_finite_01.json",),
            replicate_seeds=(11,),
            particle_count=2,
            horizon_limit=1,
            repository_root=tmp_path,
            output_path=tmp_path
            / "evilvalidation"
            / "h6_smc_accuracy.json",
        )
