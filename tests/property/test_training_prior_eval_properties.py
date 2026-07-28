from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from vfe4.data.tokenizer import (
    SyntheticTokenizerFixtureContract,
    build_synthetic_fixture_tokenizer_spec,
    encode_fixture_split_record,
    issue_fixture_split_capability,
)
from vfe4.data.windows import (
    CausalPrefix,
    build_evaluation_schedule,
    materialize_causal_window_set,
)
from vfe4.evaluation import (
    WT103EstimatorStreamBinding,
    WT103EvaluationBatches,
    bind_wt103_prior_predictor,
    score_prior_nll,
)
from vfe4.evaluation.prior_nll import wt103_estimator_stream_seed
from vfe4.predictive import EstimatorIdentity, EstimatorStream, PriorPredictor
from vfe4.types import (
    EstimatorProtocol,
    EstimatorSpec,
    VocabularyIdentity,
)


class _StopAtPrediction(RuntimeError):
    pass


class _ByteAdapter:
    distribution_name = "tiktoken"
    distribution_version = "0.12.0"
    encoding_name = "gpt2"
    vocabulary_size = 50_257
    special_tokens = (("<|endoftext|>", 50_256),)
    regex_pattern_sha256 = hashlib.sha256(b"property-regex").hexdigest()
    mergeable_ranks_sha256 = hashlib.sha256(b"property-ranks").hexdigest()
    ordinary_encoding_policy = "encode_ordinary_no_special_tokens"
    fitted_state_sha256 = None
    implementation_sha256 = hashlib.sha256(
        b"tests.task9.property-byte-adapter.v1"
    ).hexdigest()

    def encode_ordinary(self, text: str) -> tuple[int, ...]:
        return tuple(text.encode("utf-8"))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return bytes(token_ids).decode("utf-8")


class _Backend:
    def publish_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _evaluation_batches(
    tmp_path: Path,
    *,
    raw: bytes,
) -> WT103EvaluationBatches:
    adapter = _ByteAdapter()
    contract = SyntheticTokenizerFixtureContract.create(
        distribution_name=adapter.distribution_name,
        distribution_version=adapter.distribution_version,
        encoding_name=adapter.encoding_name,
        vocabulary_size=adapter.vocabulary_size,
        special_tokens=adapter.special_tokens,
        regex_pattern_sha256=adapter.regex_pattern_sha256,
        mergeable_ranks_sha256=adapter.mergeable_ranks_sha256,
        ordinary_encoding_policy=adapter.ordinary_encoding_policy,
        golden_vectors=(
            ("ascii", "abc", (97, 98, 99)),
            ("unicode", "xy", (120, 121)),
            ("newlines", "\n\n", (10, 10)),
        ),
    )
    spec = build_synthetic_fixture_tokenizer_spec(contract, adapter)
    record = encode_fixture_split_record(
        split="validation",
        raw_bytes=raw,
        raw_parent_sha256=hashlib.sha256(raw).hexdigest(),
        spec=spec,
        fixture_contract=contract,
        adapter=adapter,
        cache_root=tmp_path / "cache",
        durability_backend=_Backend(),
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    capability = issue_fixture_split_capability(
        allowed_splits=("validation",),
        cache_identities=(record,),
    )
    windows = materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_Backend(),
    )
    return WT103EvaluationBatches.create(
        windows=windows,
        schedule=build_evaluation_schedule(windows),
    )


class _PrefixProbe:
    vocabulary = VocabularyIdentity(
        "gpt2-wt103-target-blind-probe-v1",
        50_257,
        "a" * 64,
    )
    estimator_spec = EstimatorSpec.create(
        kind="deterministic_exact",
        particle_count=None,
        resampling="none",
    )
    estimator_identity = EstimatorIdentity.from_spec(estimator_spec)

    def __init__(self) -> None:
        self.observed_prefixes: list[tuple[int, ...]] = []

    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: object | None = None,
    ) -> object:
        del estimator_rng, cache
        self.observed_prefixes.append(
            tuple(int(value) for value in prefix_tokens.token_ids.tolist())
        )
        raise _StopAtPrediction


@pytest.mark.parametrize(
    ("current_target", "later_input", "later_target"),
    (
        (31, 31, 43),
        (47, 47, 71),
        (83, 83, 101),
    ),
)
def test_gpt2_current_target_and_suffix_cannot_enter_predictor_call(
    current_target: int,
    later_input: int,
    later_target: int,
    tmp_path: Path,
) -> None:
    predictor = _PrefixProbe()
    protocol = EstimatorProtocol.create()
    stream = EstimatorStream.create(
        stream_seed=wt103_estimator_stream_seed(
            split="validation",
            estimator_protocol_sha256=protocol.protocol_sha256,
            logical_stream_id=None,
        ),
        estimator_identity=predictor.estimator_identity,
    )
    binding = WT103EstimatorStreamBinding.create(
        split="validation",
        logical_stream_id=None,
        estimator_protocol=protocol,
        stream=stream,
    )
    bound_predictor = bind_wt103_prior_predictor(predictor, binding)
    assert later_input == current_target
    batches = _evaluation_batches(
        tmp_path,
        raw=bytes((17, current_target, later_target)),
    )

    with pytest.raises(_StopAtPrediction):
        score_prior_nll(
            bound_predictor,
            batches,
            stream,
        )

    assert predictor.observed_prefixes == [(17,)]


def test_public_prior_scoring_signature_has_no_target_or_recognition_channel() -> None:
    assert tuple(inspect.signature(score_prior_nll).parameters) == (
        "predictor",
        "batches",
        "stream",
    )
    assert tuple(
        inspect.signature(PriorPredictor.next_token_log_probs).parameters
    ) == ("self", "prefix_tokens", "estimator_rng", "cache")
    forbidden = {
        "target",
        "targets",
        "suffix",
        "full_window",
        "recognition",
        "posterior",
    }
    assert forbidden.isdisjoint(inspect.signature(score_prior_nll).parameters)
    assert forbidden.isdisjoint(
        inspect.signature(PriorPredictor.next_token_log_probs).parameters
    )
