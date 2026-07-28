from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest

from vfe4.training.h6_experiment_v3 import H6_CONFIRMATORY_SEEDS_V3
from vfe4.training.h6_matching_v3 import H6_MATCHING_V3_ENDPOINT_CONFIG_IDS


_A0_ENDPOINT = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0]
_COMPLETE_A5_ENDPOINT = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5]
_EMISSION_A5_ENDPOINT = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9]
_PARTICLE_COUNTS = (128, 256, 512, 1024)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _exact_row(
    module: object,
    seed: int,
    *,
    opening: str,
    endpoint: str = _A0_ENDPOINT,
) -> object:
    return module.H6ExactA0CorpusTotalV3.create(
        endpoint_config_id=endpoint,
        training_seed=seed,
        checkpoint_sha256=_sha(f"checkpoint:{endpoint}:{seed}"),
        counted_test_targets=37,
        exact_total_nll=float(seed % 1000) + 0.25,
        opening_proof_sha256=opening,
    )


def _weighted_row(
    module: object,
    *,
    role: str,
    endpoint: str,
    seed: int,
    replicate_id: int,
    particle_count: int,
    opening: str,
) -> object:
    return module.H6WeightedA5CorpusTotalV3.create(
        endpoint_role=role,
        endpoint_config_id=endpoint,
        training_seed=seed,
        checkpoint_sha256=_sha(f"checkpoint:{endpoint}:{seed}"),
        particle_count=particle_count,
        replicate_id=replicate_id,
        counted_test_targets=37,
        weighted_total_nll=(
            float(seed % 1000) + replicate_id / 100.0 + particle_count / 10000.0
        ),
        monte_carlo_half_width=0.125,
        smc_bias_bound=0.0625,
        opening_proof_sha256=opening,
    )


@lru_cache(maxsize=1)
def _rows() -> tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]:
    import vfe4.artifacts.h6_prediction_v3 as artifacts

    opening = _sha("opening")
    exact = tuple(
        _exact_row(artifacts, seed, opening=opening)
        for seed in H6_CONFIRMATORY_SEEDS_V3
    )
    complete = tuple(
        _weighted_row(
            artifacts,
            role="complete_a5",
            endpoint=_COMPLETE_A5_ENDPOINT,
            seed=seed,
            replicate_id=replicate_id,
            particle_count=particle_count,
            opening=opening,
        )
        for seed in H6_CONFIRMATORY_SEEDS_V3
        for replicate_id in range(64)
        for particle_count in _PARTICLE_COUNTS
    )
    emission = tuple(
        _weighted_row(
            artifacts,
            role="emission_a5",
            endpoint=_EMISSION_A5_ENDPOINT,
            seed=seed,
            replicate_id=replicate_id,
            particle_count=particle_count,
            opening=opening,
        )
        for seed in H6_CONFIRMATORY_SEEDS_V3
        for replicate_id in range(64)
        for particle_count in _PARTICLE_COUNTS
    )
    return exact, complete, emission


def test_raw_inventory_accepts_exactly_4104_discriminated_rows() -> None:
    import vfe4.artifacts.h6_prediction_v3 as artifacts

    exact, complete, emission = _rows()
    inventory = artifacts.H6RawEndpointInventoryV4.create(
        exact_a0_rows=exact,
        complete_a5_rows=complete,
        emission_a5_rows=emission,
    )

    assert inventory.logical_row_count == 4104
    assert len(inventory.exact_a0_rows) == 8
    assert len(inventory.complete_a5_rows) == 2048
    assert len(inventory.emission_a5_rows) == 2048
    with pytest.raises(ValueError, match="4104"):
        artifacts.H6RawEndpointInventoryV4.from_rows((exact[0],) * 24576)
    with pytest.raises(ValueError, match="duplicate"):
        artifacts.H6RawEndpointInventoryV4.create(
            exact_a0_rows=(exact[0],) * 8,
            complete_a5_rows=complete,
            emission_a5_rows=emission,
        )
    with pytest.raises(ValueError, match="opening"):
        artifacts.H6RawEndpointInventoryV4.create(
            exact_a0_rows=(
                *exact[:-1],
                _exact_row(
                    artifacts,
                    H6_CONFIRMATORY_SEEDS_V3[-1],
                    opening=_sha("another-opening"),
                ),
            ),
            complete_a5_rows=complete,
            emission_a5_rows=emission,
        )


def test_weighted_a5_rows_require_frozen_particles_replicates_and_common_streams() -> (
    None
):
    _, complete, emission = _rows()
    with pytest.raises(ValueError, match="particle"):
        replace(complete[0], particle_count=64)
    with pytest.raises(ValueError, match="replicate"):
        replace(complete[0], replicate_id=64)
    with pytest.raises(ValueError, match="common stream"):
        replace(emission[0], common_stream_sha256=_sha("different-stream"))


def test_exact_a0_rows_reject_every_weighted_estimator_field() -> None:
    import vfe4.artifacts.h6_prediction_v3 as artifacts

    opening = _sha("opening")
    seed = H6_CONFIRMATORY_SEEDS_V3[0]
    with pytest.raises(TypeError):
        artifacts.H6ExactA0CorpusTotalV3.create(
            endpoint_config_id=_A0_ENDPOINT,
            training_seed=seed,
            checkpoint_sha256=_sha("checkpoint"),
            counted_test_targets=37,
            exact_total_nll=1.0,
            opening_proof_sha256=opening,
            particle_count=128,
        )
    with pytest.raises(ValueError, match="A0"):
        _exact_row(
            artifacts,
            seed,
            opening=opening,
            endpoint=_COMPLETE_A5_ENDPOINT,
        )
    with pytest.raises(ValueError, match="weighted A5"):
        _weighted_row(
            artifacts,
            role="complete_a5",
            endpoint=_A0_ENDPOINT,
            seed=seed,
            replicate_id=0,
            particle_count=128,
            opening=opening,
        )


def test_complete_a5_rows_are_reused_without_rescoring() -> None:
    import vfe4.artifacts.h6_prediction_v3 as artifacts

    exact, complete, emission = _rows()
    inventory = artifacts.H6RawEndpointInventoryV4.create(
        exact_a0_rows=exact,
        complete_a5_rows=complete,
        emission_a5_rows=emission,
    )
    metrics = artifacts.H6PredictionMetricsV3.from_raw_inventory(inventory)

    assert (
        metrics.primary_complete_a5_row_sha256s
        is metrics.objective_complete_a5_row_sha256s
    )
    assert metrics.primary_complete_a5_row_sha256s == tuple(
        row.row_sha256 for row in complete
    )
    assert len(inventory.complete_a5_rows) == 2048


def test_weighted_common_streams_are_frozen_reused_and_not_caller_selected() -> None:
    import vfe4.artifacts.h6_prediction_v3 as artifacts

    _, complete, emission = _rows()
    complete_streams = tuple(row.common_stream_sha256 for row in complete)
    emission_streams = tuple(row.common_stream_sha256 for row in emission)

    assert complete_streams == emission_streams
    assert len(set(complete_streams)) == 64
    assert complete_streams[:4] == (complete_streams[0],) * 4
    assert complete_streams[0] == complete_streams[256]
    assert complete[0].common_stream_sha256 == (
        artifacts.h6_weighted_common_stream_sha256_v3(
            replicate_id=0,
        )
    )
    assert (
        artifacts.H6_WEIGHTED_COMMON_STREAM_ROOT_SEED,
        artifacts.H6_WEIGHTED_COMMON_STREAM_DOMAIN,
    ) == (2026072198, "h6-wt2-endpoint-mc-v1")


def test_raw_row_discriminator_literals_are_exact_and_hashed() -> None:
    exact, complete, _ = _rows()

    assert exact[0].row_kind == "exact_a0_corpus_total"
    assert complete[0].row_kind == "weighted_a5_smc_corpus_total"
    with pytest.raises(ValueError, match="discriminator|row kind"):
        replace(exact[0], row_kind="weighted_a5_smc_corpus_total")
    with pytest.raises(ValueError, match="discriminator|row kind"):
        replace(complete[0], row_kind="exact_a0_corpus_total")


def test_prediction_result_v3_is_typed_hashed_and_round_trips(
    tmp_path: Path,
) -> None:
    import vfe4.artifacts.h6_prediction_v3 as artifacts

    exact, complete, emission = _rows()
    inventory = artifacts.H6RawEndpointInventoryV4.create(
        exact_a0_rows=exact,
        complete_a5_rows=complete,
        emission_a5_rows=emission,
    )
    metrics = artifacts.H6PredictionMetricsV3.from_raw_inventory(inventory)
    result = artifacts.H6PredictionResultV3.create(
        reservation_sha256=_sha("reservation"),
        opening_proof_sha256=_sha("opening"),
        inventory=inventory,
        metrics=metrics,
    )

    published = artifacts.publish_h6_prediction_result_v3(
        tmp_path,
        "result",
        result=result,
        inventory=inventory,
        metrics=metrics,
    )
    reopened = artifacts.read_h6_prediction_result_v3(
        published,
        expected_result_sha256=result.result_sha256,
    )

    assert result.result_schema == "h6-prediction-result-v3"
    assert reopened == (result, inventory, metrics)
    with pytest.raises(Exception, match="exists|replace|refus"):
        artifacts.publish_h6_prediction_result_v3(
            tmp_path,
            "result",
            result=result,
            inventory=inventory,
            metrics=metrics,
        )


def test_task10_artifact_exports_are_exact_and_lazy() -> None:
    import vfe4.artifacts as public_artifacts
    import vfe4.artifacts.h6_prediction_v3 as artifacts

    names = {
        "H6ExactA0CorpusTotalV3",
        "H6WeightedA5CorpusTotalV3",
        "H6RawEndpointInventoryV4",
        "H6PredictionMetricsV3",
        "H6PredictionResultV3",
        "H6_WEIGHTED_COMMON_STREAM_ROOT_SEED",
        "H6_WEIGHTED_COMMON_STREAM_DOMAIN",
        "H6_WEIGHTED_COMMON_STREAM_REGISTRY_SHA256",
        "h6_weighted_common_stream_sha256_v3",
        "publish_h6_prediction_result_v3",
        "read_h6_prediction_result_v3",
    }
    assert names <= set(artifacts.__all__)
    assert names <= set(public_artifacts.__all__)
    for name in names:
        assert getattr(public_artifacts, name) is getattr(artifacts, name)
