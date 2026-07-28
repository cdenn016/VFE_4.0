from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from vfe4.artifacts.durability import (
    DurabilityCollisionError,
    DurableFileIdentity,
)
from vfe4.data.windows import (
    CausalWindowSet,
    WindowManifest,
    build_evaluation_schedule,
    enumerate_wt103_window_rows,
    iter_causal_batches,
    materialize_causal_window_set,
)
from vfe4.data.tokenizer import (
    SyntheticTokenizerFixtureContract,
    build_synthetic_fixture_tokenizer_spec,
    encode_fixture_split_record,
    issue_fixture_split_capability,
)
import vfe4.evaluation.test_opening as test_opening_module
import vfe4.evaluation.statistics as statistics_module
import vfe4.predictive.prior as predictive_prior_module
from vfe4.evaluation.prior_nll import (
    WT103ScoreTrace,
    wt103_common_stream_registry_sha256,
    wt103_estimator_stream_seed,
    wt103_score_trace,
)
from vfe4.evaluation import (
    DurableTestOpeningCapability,
    TestOpeningPlan as WT103TestOpeningPlan,
    WT103EstimatorStreamBinding,
    WT103EvaluationBatches,
    WT103RawScoreRecord,
    aggregate_a5_smc,
    bind_wt103_prior_predictor,
    paired_prediction_decision,
    reserve_test_opening,
    score_prior_nll,
)
from vfe4.generative import FixedSourcePrior, LanguageGenerativeModel
from vfe4.predictive import (
    BootstrapSmcPredictor,
    EstimatorIdentity,
    EstimatorRecord,
    EstimatorStream,
    LanguageGenerativeProposalAdapter,
)
from vfe4.types import (
    CausalDag,
    CausalDagRow,
    EndpointInventory,
    EstimatorProtocol,
    EstimatorSpec,
    H6LanguageStructure,
    WT103CheckpointIdentity,
    WT103EvaluationRecord,
    WT103NllTotals,
    VocabularyIdentity,
    ZeroDimensionalBase,
    default_wt103_arm_specs,
    default_wt103_gate_specs,
)
from vfe4.types.results import GateStatus
from vfe4.types.training import (
    WT103_CONFIRMATORY_SEED_IDS,
    WT103_PARTICLE_COUNTS,
    WT103_TEST_STREAM_IDS,
    WT103_TUNING_CELLS,
    WT103_TUNING_SEED_IDS,
    owned_sha256,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


class _TinyTokenizerAdapter:
    distribution_name = "tiktoken"
    distribution_version = "0.12.0"
    encoding_name = "gpt2"
    vocabulary_size = 50_257
    special_tokens = (("<|endoftext|>", 50_256),)
    regex_pattern_sha256 = hashlib.sha256(b"task9-regex").hexdigest()
    mergeable_ranks_sha256 = hashlib.sha256(b"task9-ranks").hexdigest()
    ordinary_encoding_policy = "encode_ordinary_no_special_tokens"
    fitted_state_sha256 = None
    implementation_sha256 = hashlib.sha256(
        b"tests.task9.tiny-tokenizer-adapter.v1"
    ).hexdigest()
    _encode = {"a": 0, "b": 1, "c": 2}
    _decode = ("a", "b", "c")

    def encode_ordinary(self, text: str) -> tuple[int, ...]:
        return tuple(self._encode[character] for character in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(self._decode[token_id] for token_id in token_ids)


class _FixtureBackend:
    def publish_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _validation_windows(
    tmp_path: Path,
    *,
    raw: bytes = b"abca",
) -> CausalWindowSet:
    adapter = _TinyTokenizerAdapter()
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
            ("ascii", "abc", (0, 1, 2)),
            ("unicode", "cab", (2, 0, 1)),
            ("newlines", "bca", (1, 2, 0)),
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
        durability_backend=_FixtureBackend(),
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    capability = issue_fixture_split_capability(
        allowed_splits=("validation",),
        cache_identities=(record,),
    )
    return materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_FixtureBackend(),
    )


def _typed_evaluation_batches(
    tmp_path: Path,
    *,
    raw: bytes = b"abca",
) -> WT103EvaluationBatches:
    windows = _validation_windows(tmp_path, raw=raw)
    return WT103EvaluationBatches.create(
        windows=windows,
        schedule=build_evaluation_schedule(windows),
    )


def _sealed_test_windows(tmp_path: Path) -> CausalWindowSet:
    adapter = _TinyTokenizerAdapter()
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
            ("ascii", "abc", (0, 1, 2)),
            ("unicode", "cab", (2, 0, 1)),
            ("newlines", "bca", (1, 2, 0)),
        ),
    )
    spec = build_synthetic_fixture_tokenizer_spec(contract, adapter)
    raw = b"abca"
    record = encode_fixture_split_record(
        split="test",
        raw_bytes=raw,
        raw_parent_sha256=hashlib.sha256(raw).hexdigest(),
        spec=spec,
        fixture_contract=contract,
        adapter=adapter,
        cache_root=tmp_path / "cache",
        durability_backend=_FixtureBackend(),
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    rows = enumerate_wt103_window_rows(record.token_count)
    row_payload = b"VFE4-WT103-WINDOW-ROWS-V1\x00" + b"".join(
        row.canonical_bytes() for row in rows
    )
    manifest = WindowManifest.create(
        split="test",
        token_payload_sha256=record.payload_sha256,
        window_count=len(rows),
        counted_targets=record.token_count - 1,
        payload_sha256=hashlib.sha256(row_payload).hexdigest(),
    )
    token_path = tmp_path / "cache" / record.cache_relative_path
    tokens = np.memmap(
        token_path,
        mode="r",
        dtype=np.dtype("<i4"),
        shape=(record.token_count,),
    )
    return CausalWindowSet(
        split="test",
        cache_record=record,
        tokenizer_spec=spec,
        manifest=manifest,
        rows=rows,
        row_payload_relative_path=(
            f"window-manifests/test/{manifest.payload_sha256}.rows"
        ),
        token_payload_path=token_path,
        _tokens=tokens,
    )


class _RawOpeningBackend:
    def __init__(self) -> None:
        self.payloads: dict[Path, bytes] = {}

    def probe(self, root: Path) -> object:
        del root
        return object()

    def create_exclusive(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if path in self.payloads:
            raise DurabilityCollisionError("reservation already exists")
        self.payloads[path] = payload
        return DurableFileIdentity.create(
            operation="exclusive_create",
            payload=payload,
            volume_identity="task9-raw-fixture-volume",
        )

    def replace_durable(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if path not in self.payloads:
            raise RuntimeError("replacement target is absent")
        self.payloads[path] = payload
        return DurableFileIdentity.create(
            operation="replace",
            payload=payload,
            volume_identity="task9-raw-fixture-volume",
        )

    def publish_bytes(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if path in self.payloads:
            return self.replace_durable(path, payload)
        return self.create_exclusive(path, payload)


@dataclass(frozen=True, slots=True)
class _RawFixtureContext:
    opening_capability: DurableTestOpeningCapability
    evaluation_batches: WT103EvaluationBatches


def _raw_fixture_context(
    tmp_path: Path,
    inventory: EndpointInventory,
) -> _RawFixtureContext:
    windows = _sealed_test_windows(tmp_path)
    schedule = build_evaluation_schedule(windows)
    backend = _RawOpeningBackend()
    checkpoints = tuple(
        _checkpoint(key) for key in inventory.terminal_checkpoint_keys
    )
    plan = WT103TestOpeningPlan.create(
        repository_root=tmp_path,
        durability_backend=backend,
        endpoint_inventory=inventory,
        terminal_checkpoints=checkpoints,
        run_group_complete=True,
        run_group_manifest_sha256="1" * 64,
        analysis_sha256="2" * 64,
        figure_sha256="3" * 64,
        data_identity_sha256=windows.cache_record.raw_parent_sha256,
        tokenizer_identity_sha256=windows.tokenizer_spec.spec_sha256,
        test_window_manifest_sha256=windows.manifest.manifest_sha256,
        test_schedule_sha256=schedule.schedule_sha256,
    )
    capability = reserve_test_opening(plan)
    opened = test_opening_module._unseal_test_windows(  # noqa: SLF001
        capability,
        lambda: windows,
    )
    evaluation_batches = WT103EvaluationBatches.create(
        windows=opened,
        schedule=schedule,
    )
    return _RawFixtureContext(
        opening_capability=capability,
        evaluation_batches=evaluation_batches,
    )


def _inventory() -> EndpointInventory:
    return EndpointInventory.create(
        default_wt103_arm_specs(),
        default_wt103_gate_specs(),
        WT103_TUNING_CELLS,
        WT103_TUNING_SEED_IDS,
        WT103_CONFIRMATORY_SEED_IDS,
        EstimatorProtocol.create(),
    )


def _small_model() -> LanguageGenerativeModel:
    dag = CausalDag.create(
        node_labels=(0, 1, 2, 3),
        rows=tuple(
            CausalDagRow(receiver_t, tuple(range(receiver_t)))
            for receiver_t in range(1, 4)
        ),
    )
    structure = H6LanguageStructure.create(
        base=ZeroDimensionalBase.create(),
        dag=dag,
        receiver_labels=(1, 2, 3),
    )
    vocabulary = VocabularyIdentity("wt103-eval-small-v1", 3, _SHA_A)
    source_rows = tuple(
        torch.linspace(-0.2, 0.2, receiver_t, dtype=torch.float64)
        for receiver_t in range(1, 4)
    )
    source_prior = FixedSourcePrior(
        structure=structure,
        vocabulary=vocabulary,
        fixture_sha256=_SHA_A,
        predictor_config_sha256=_SHA_B,
        model_family_sha256=_SHA_C,
        state_logits=source_rows,
        model_logits=tuple(row.flip(0) for row in source_rows),
    )
    model = LanguageGenerativeModel(
        structure=structure,
        vocabulary=vocabulary,
        model_family_sha256=_SHA_C,
        latent_dim=2,
        source_prior=source_prior,
    )
    with torch.no_grad():
        model.initial_log_scale.fill_(-0.4)
        model.model_transition_weight.copy_(
            torch.tensor([[0.65, 0.05], [-0.1, 0.55]], dtype=torch.float64)
        )
        model.state_transition_weight.copy_(
            torch.tensor([[0.5, 0.1], [0.05, 0.6]], dtype=torch.float64)
        )
        model.state_model_weight.copy_(
            torch.tensor([[0.2, 0.0], [0.0, -0.15]], dtype=torch.float64)
        )
        model.model_transition_log_scale.fill_(-0.7)
        model.state_transition_log_scale.fill_(-0.6)
        model.emission_state_weight.copy_(
            torch.tensor(
                [[0.25, -0.1], [-0.15, 0.2], [0.05, 0.05]],
                dtype=torch.float64,
            )
        )
        model.emission_model_weight.copy_(
            torch.tensor(
                [[0.1, 0.05], [0.0, -0.1], [-0.1, 0.05]],
                dtype=torch.float64,
            )
        )
        model.emission_bias.copy_(
            torch.tensor([0.08, -0.03, -0.05], dtype=torch.float64)
        )
    return model


def _raw_predictor_and_stream() -> tuple[
    BootstrapSmcPredictor,
    EstimatorStream,
]:
    model = _small_model()
    spec = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=128,
        resampling="systematic_ess_half",
    )
    identity = EstimatorIdentity.from_spec(spec)
    predictor = BootstrapSmcPredictor(
        proposal=LanguageGenerativeProposalAdapter(model),
        estimator_spec=spec,
        estimator_identity=identity,
        predictor_config_sha256=_SHA_B,
        data_safety_sha256=_SHA_D,
    )
    stream = EstimatorStream.create(
        stream_seed=9_918_273,
        estimator_identity=identity,
    )
    return predictor, stream


def _predictor_and_stream() -> tuple[object, EstimatorStream]:
    predictor, _ = _raw_predictor_and_stream()
    protocol = EstimatorProtocol.create()
    stream = EstimatorStream.create(
        stream_seed=wt103_estimator_stream_seed(
            split="validation",
            estimator_protocol_sha256=protocol.protocol_sha256,
            logical_stream_id=0,
        ),
        estimator_identity=predictor.estimator_identity,
    )
    binding = WT103EstimatorStreamBinding.create(
        split="validation",
        logical_stream_id=0,
        estimator_protocol=protocol,
        stream=stream,
    )
    return bind_wt103_prior_predictor(predictor, binding), stream


def test_score_prior_nll_uses_target_blind_corpus_sums_and_cache_audit(
    tmp_path: Path,
) -> None:
    predictor, stream = _predictor_and_stream()
    evaluation_batches = _typed_evaluation_batches(tmp_path)
    batches = evaluation_batches.batches

    totals = score_prior_nll(
        predictor,
        evaluation_batches,
        stream,
    )

    assert type(totals) is WT103NllTotals
    assert totals.scorer_kind == "weighted_smc"
    assert totals.particle_count == 128
    assert totals.estimator_stream_id == 0
    assert totals.estimator_stream_id != stream.stream_seed
    assert totals.counted_targets == 3
    assert totals.nll_per_token == totals.summed_nll / 3
    assert totals.perplexity == math.exp(totals.nll_per_token)
    assert len(totals.cache_audit_sha256) == 64

    per_target = []
    for batch in batches:
        for row_index in range(len(batch.window_ids)):
            for position in range(
                int(torch.sum(batch.attention_mask[row_index]).item())
            ):
                per_target.append(
                    float(batch.targets[row_index, position].item())
                )
    assert len(per_target) == totals.counted_targets


def test_score_prior_nll_requires_manifest_bound_complete_batches(
    tmp_path: Path,
) -> None:
    predictor, stream = _predictor_and_stream()
    evaluation_batches = _typed_evaluation_batches(tmp_path)
    with pytest.raises(TypeError):
        WT103EvaluationBatches.create(
            manifest=evaluation_batches.manifest,  # type: ignore[call-arg]
            batches=evaluation_batches.batches,  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="manifest-bound"):
        score_prior_nll(
            predictor,
            evaluation_batches.batches,
            stream,
        )


def test_evaluation_batches_reject_caller_signed_corpus_substitution(
    tmp_path: Path,
) -> None:
    windows = _validation_windows(tmp_path)
    schedule = build_evaluation_schedule(windows)
    evaluation = WT103EvaluationBatches.create(
        windows=windows,
        schedule=schedule,
    )
    forged = tuple(iter_causal_batches(windows=windows, schedule=schedule))
    forged[0].targets[0, 0] = 2

    with pytest.raises(
        ValueError,
        match="shifted causal data|window source",
    ):
        replace(evaluation, batches=forged)


def test_stream_binding_rejects_noncanonical_counter_seed() -> None:
    predictor, stream = _raw_predictor_and_stream()

    with pytest.raises(ValueError, match="canonical counter seed"):
        WT103EstimatorStreamBinding.create(
            split="validation",
            logical_stream_id=0,
            estimator_protocol=EstimatorProtocol.create(),
            stream=stream,
        )

    assert predictor.estimator_identity.identity_sha256 == (
        stream.estimator_identity_sha256
    )


def test_wt103_stream_seed_mapping_is_canonical_and_frozen() -> None:
    protocol = EstimatorProtocol.create()
    assert tuple(
        wt103_estimator_stream_seed(
            split="validation",
            estimator_protocol_sha256=protocol.protocol_sha256,
            logical_stream_id=stream_id,
        )
        for stream_id in (0, 1, 7)
    ) == (
        373_636_910_170_320_889,
        9_718_098_125_836_363_378,
        10_160_547_806_302_829_984,
    )
    assert tuple(
        wt103_estimator_stream_seed(
            split="test",
            estimator_protocol_sha256=protocol.protocol_sha256,
            logical_stream_id=stream_id,
        )
        for stream_id in (0, 1, 63)
    ) == (
        1_033_995_588_453_967_296,
        16_657_823_452_050_025_941,
        5_544_989_406_361_717_292,
    )
    assert wt103_common_stream_registry_sha256(
        split="test",
        estimator_protocol_sha256=protocol.protocol_sha256,
        logical_stream_id=0,
    ) == "eb9257d8b77e543860168bee9d32c4121eb54bf3001f4b047978c30f1bdafc00"
    assert statistics_module._common_test_stream_registry_sha256(  # noqa: SLF001
        estimator_protocol_sha256=protocol.protocol_sha256,
        stream_id=0,
    ) == "eb9257d8b77e543860168bee9d32c4121eb54bf3001f4b047978c30f1bdafc00"


def test_target_perturbation_changes_score_but_not_raw_predictions(
    tmp_path: Path,
) -> None:
    predictor, stream = _predictor_and_stream()
    left = _typed_evaluation_batches(tmp_path / "left", raw=b"abca")
    right = _typed_evaluation_batches(tmp_path / "right", raw=b"abcb")

    left_totals = score_prior_nll(
        predictor,
        left,
        stream,
    )
    right_totals = score_prior_nll(
        predictor,
        right,
        stream,
    )

    assert left_totals.summed_nll != right_totals.summed_nll
    assert tuple(
        record.record_sha256
        for record in wt103_score_trace(left_totals).estimator_records
    ) == tuple(
        record.record_sha256
        for record in wt103_score_trace(right_totals).estimator_records
    )
    assert left_totals.cache_audit_sha256 != right_totals.cache_audit_sha256


class _CacheOrderSensitivePredictor:
    def __init__(
        self,
        predictor: BootstrapSmcPredictor,
        alternate_stream: EstimatorStream,
    ) -> None:
        self._predictor = predictor
        self._alternate_stream = alternate_stream
        self.vocabulary = predictor.vocabulary
        self.estimator_spec = predictor.estimator_spec
        self.estimator_identity = predictor.estimator_identity

    def next_token_log_probs(
        self,
        prefix_tokens: object,
        estimator_rng: EstimatorStream,
        cache: object | None = None,
    ) -> object:
        if cache is not None:
            return self._predictor.next_token_log_probs(
                prefix_tokens,  # type: ignore[arg-type]
                self._alternate_stream,
                None,
            )
        return self._predictor.next_token_log_probs(
            prefix_tokens,  # type: ignore[arg-type]
            estimator_rng,
            None,
        )


def test_cache_order_sensitive_predictor_is_rejected(
    tmp_path: Path,
) -> None:
    predictor, _ = _raw_predictor_and_stream()
    protocol = EstimatorProtocol.create()
    stream = EstimatorStream.create(
        stream_seed=wt103_estimator_stream_seed(
            split="validation",
            estimator_protocol_sha256=protocol.protocol_sha256,
            logical_stream_id=0,
        ),
        estimator_identity=predictor.estimator_identity,
    )
    alternate = EstimatorStream.create(
        stream_seed=stream.stream_seed + 1,
        estimator_identity=predictor.estimator_identity,
    )
    sensitive = _CacheOrderSensitivePredictor(predictor, alternate)
    binding = WT103EstimatorStreamBinding.create(
        split="validation",
        logical_stream_id=0,
        estimator_protocol=protocol,
        stream=stream,
    )
    bound = bind_wt103_prior_predictor(sensitive, binding)
    with pytest.raises(ValueError, match="cold, warm, and reverse"):
        score_prior_nll(
            bound,
            _typed_evaluation_batches(tmp_path),
            stream,
        )


def _checkpoint(
    logical_key: str,
    *,
    role: str = "terminal_scoring",
) -> WT103CheckpointIdentity:
    payload_sha256 = hashlib.sha256(
        f"{logical_key}|payload".encode("ascii")
    ).hexdigest()
    manifest_sha256 = hashlib.sha256(
        f"{logical_key}|manifest".encode("ascii")
    ).hexdigest()
    artifact_sha256 = hashlib.sha256(
        b"vfe4-checkpoint-artifact-v1\x00"
        + bytes.fromhex(payload_sha256)
        + bytes.fromhex(manifest_sha256)
    ).hexdigest()
    payload = {
        "schema_version": "wt103-checkpoint-identity-v1",
        "logical_key": logical_key,
        "checkpoint_role": role,
        "scientific_state_sha256": hashlib.sha256(
            f"{logical_key}|scientific".encode("ascii")
        ).hexdigest(),
        "checkpoint_payload_sha256": payload_sha256,
        "checkpoint_manifest_body_sha256": manifest_sha256,
        "artifact_sha256": artifact_sha256,
        "size_bytes": 1,
    }
    return WT103CheckpointIdentity(
        **payload,
        checkpoint_identity_sha256=owned_sha256(
            "vfe4.wt103.checkpoint-identity.v1",
            payload,
        ),
    )


def _totals(
    *,
    scorer_kind: str,
    nll_per_token: float,
    counted_targets: int,
    stream_id: int | None,
    particle_count: int | None,
    key: str,
) -> WT103NllTotals:
    summed_nll = float(
        math.fsum(nll_per_token for _ in range(counted_targets))
    )
    derived = summed_nll / counted_targets
    payload = {
        "schema_version": "wt103-nll-totals-v1",
        "scorer_kind": scorer_kind,
        "summed_nll": summed_nll,
        "counted_targets": counted_targets,
        "nll_per_token": derived,
        "perplexity": math.exp(derived),
        "estimator_stream_id": stream_id,
        "particle_count": particle_count,
        "cache_audit_sha256": hashlib.sha256(
            f"{key}|cache-audit".encode("ascii")
        ).hexdigest(),
    }
    return WT103NllTotals(
        **payload,
        totals_sha256=owned_sha256(
            "vfe4.wt103.nll-totals.v1",
            payload,
        ),
    )


def _evaluation_record(
    *,
    endpoint_key: str,
    checkpoint: WT103CheckpointIdentity,
    totals: WT103NllTotals,
) -> WT103EvaluationRecord:
    payload = {
        "schema_version": "wt103-evaluation-record-v1",
        "endpoint_key": endpoint_key,
        "checkpoint": checkpoint,
        "totals": totals,
        "target_blind_cache_audit_passed": True,
    }
    return WT103EvaluationRecord(
        **payload,
        evaluation_sha256=owned_sha256(
            "vfe4.wt103.evaluation-record.v1",
            payload,
        ),
    )


def _raw_score_record(
    *,
    inventory: EndpointInventory,
    evaluation: WT103EvaluationRecord,
    context: _RawFixtureContext,
) -> WT103RawScoreRecord:
    totals = evaluation.totals
    weighted = totals.scorer_kind == "weighted_smc"
    spec = EstimatorSpec.create(
        kind="weighted_smc" if weighted else "deterministic_exact",
        particle_count=totals.particle_count,
        resampling=(
            "systematic_ess_half" if weighted else "none"
        ),
    )
    identity = EstimatorIdentity.from_spec(spec)
    protocol = EstimatorProtocol.create()
    stream = EstimatorStream.create(
        stream_seed=wt103_estimator_stream_seed(
            split="test",
            estimator_protocol_sha256=protocol.protocol_sha256,
            logical_stream_id=totals.estimator_stream_id,
        ),
        estimator_identity=identity,
    )
    binding = WT103EstimatorStreamBinding.create(
        split="test",
        logical_stream_id=totals.estimator_stream_id,
        estimator_protocol=protocol,
        stream=stream,
    )
    estimator_records: list[EstimatorRecord] = []
    for index in range(totals.counted_targets):
        record_payload = {
            "estimator_semantic_sha256": (
                stream.estimator_semantic_sha256
            ),
            "estimator_artifact_bytes_sha256": (
                stream.estimator_artifact_bytes_sha256
            ),
            "estimator_stream_sha256": stream.stream_sha256,
            "stream_seed": stream.stream_seed,
            "prefix_sha256": hashlib.sha256(
                f"task9-fixture-prefix-{index}".encode("ascii")
            ).hexdigest(),
            "counter_trace_sha256": hashlib.sha256(
                (
                    f"task9-fixture-counter-{totals.particle_count}-"
                    f"{totals.estimator_stream_id}-{index}"
                ).encode("ascii")
            ).hexdigest(),
            "counter_draw_count": (
                totals.particle_count if weighted else 0
            ),
            "cumulative_log_normalizer": 0.0,
        }
        estimator_records.append(
            EstimatorRecord(
                **record_payload,
                record_sha256=predictive_prior_module._owned_hash(  # noqa: SLF001
                    "vfe4.h6.estimator-record.v1",
                    record_payload,
                ),
            )
        )
    score_trace = WT103ScoreTrace.create(
        evaluation_batches=context.evaluation_batches,
        binding=binding,
        stream=stream,
        totals=totals,
        estimator_records=tuple(estimator_records),
        negative_log_terms=tuple(
            totals.nll_per_token for _ in range(totals.counted_targets)
        ),
    )
    return WT103RawScoreRecord.create_finalized(
        inventory=inventory,
        evaluation=evaluation,
        opening_capability=context.opening_capability,
        evaluation_batches=context.evaluation_batches,
        score_trace=score_trace,
    )


def _complete_raw_records(
    inventory: EndpointInventory,
    tmp_path: Path,
    *,
    ineligible_control_arm: str | None = None,
    base_nll_overrides: dict[str, float] | None = None,
    context: _RawFixtureContext | None = None,
) -> tuple[WT103RawScoreRecord, ...]:
    base_nll = {
        "WT103-A0-AR-v1": 1.200,
        "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1": 1.180,
        "WT103-A5-FIXED-COMPLETE-v1": 1.190,
        "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1": 1.175,
        "WT103-A5-NOLATENT-v1": 1.210,
    }
    if base_nll_overrides is not None:
        base_nll.update(base_nll_overrides)
    exact_context = (
        _raw_fixture_context(tmp_path, inventory)
        if context is None
        else context
    )
    counted_targets = (
        exact_context.evaluation_batches.manifest.counted_targets
    )
    records: list[WT103RawScoreRecord] = []
    for arm in inventory.arms:
        for seed_index, seed in enumerate(inventory.confirmatory_seed_ids):
            terminal_key = f"terminal/{arm.arm_id}/seed={seed}"
            checkpoint = _checkpoint(terminal_key)
            test_endpoint = f"test/{terminal_key}"
            seed_shift = seed_index * 1.0e-5
            if arm.scorer_kind == "exact_autoregressive":
                raw_key = f"raw-score/test/{test_endpoint}/exact"
                evaluation = _evaluation_record(
                    endpoint_key=raw_key,
                    checkpoint=checkpoint,
                    totals=_totals(
                        scorer_kind="exact_autoregressive",
                        nll_per_token=base_nll[arm.arm_id] + seed_shift,
                        counted_targets=counted_targets,
                        stream_id=None,
                        particle_count=None,
                        key=raw_key,
                    ),
                )
                records.append(
                    _raw_score_record(
                        inventory=inventory,
                        evaluation=evaluation,
                        context=exact_context,
                    )
                )
                continue
            for particle_count in WT103_PARTICLE_COUNTS:
                for stream_id in WT103_TEST_STREAM_IDS:
                    raw_key = (
                        f"raw-score/test/{test_endpoint}/"
                        f"particles={particle_count}/stream={stream_id}"
                    )
                    stream_shift = (stream_id - 31.5) * 1.0e-6
                    ineligible_shift = (
                        0.01
                        if (
                            arm.arm_id == ineligible_control_arm
                            and particle_count == 1024
                        )
                        else 0.0
                    )
                    evaluation = _evaluation_record(
                        endpoint_key=raw_key,
                        checkpoint=checkpoint,
                        totals=_totals(
                            scorer_kind="weighted_smc",
                            nll_per_token=(
                                base_nll[arm.arm_id]
                                + seed_shift
                                + stream_shift
                                + ineligible_shift
                            ),
                            counted_targets=counted_targets,
                            stream_id=stream_id,
                            particle_count=particle_count,
                            key=raw_key,
                        ),
                    )
                    records.append(
                        _raw_score_record(
                            inventory=inventory,
                            evaluation=evaluation,
                            context=exact_context,
                        )
                    )
    return tuple(records)


def test_raw_record_factory_requires_typed_upstream_evidence() -> None:
    parameters = inspect.signature(
        WT103RawScoreRecord.create_finalized
    ).parameters
    assert {
        "opening_capability",
        "evaluation_batches",
        "score_trace",
    }.issubset(parameters)
    assert {
        "opening_capability_sha256",
        "reservation_identity_sha256",
        "data_identity_sha256",
        "tokenizer_identity_sha256",
        "window_manifest_sha256",
        "schedule_sha256",
        "estimator_stream_binding_sha256",
        "estimator_stream_sha256",
        "counter_trace_sha256",
        "counter_draw_count",
    }.isdisjoint(parameters)


def test_statistics_require_complete_inventory_and_enumerate_all_corners(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    records = _complete_raw_records(inventory, tmp_path)
    exact_raw = records[0]
    weighted_raw = next(
        record
        for record in records
        if record.scorer_kind == "weighted_smc"
    )
    assert exact_raw.endpoint_inventory_sha256 == (
        inventory.endpoint_inventory_sha256
    )
    assert exact_raw.estimator_stream_sha256 is None
    assert exact_raw.counter_trace_sha256 is None
    assert exact_raw.counter_draw_count is None
    assert weighted_raw.estimator_stream_binding_sha256 is not None
    assert weighted_raw.estimator_stream_sha256 is not None
    assert weighted_raw.common_stream_registry_sha256 is not None
    assert weighted_raw.counter_trace_sha256 is not None
    assert weighted_raw.counter_draw_count > 0
    common_stream_records = tuple(
        record
        for record in records
        if record.scorer_kind == "weighted_smc"
        and record.checkpoint == weighted_raw.checkpoint
        and record.logical_stream_id == weighted_raw.logical_stream_id
    )
    assert len(common_stream_records) == len(WT103_PARTICLE_COUNTS)
    assert len({record.stream_seed for record in common_stream_records}) == 1
    assert len(
        {
            record.common_stream_registry_sha256
            for record in common_stream_records
        }
    ) == 1
    with pytest.raises(TypeError):
        replace(weighted_raw, estimator_stream_sha256="f" * 64)

    aggregation = aggregate_a5_smc(
        records,
        inventory=inventory,
    )
    result = paired_prediction_decision(
        records,
        inventory=inventory,
    )

    assert aggregation.status is GateStatus.PASS
    assert len(aggregation.seed_estimates) == 40
    weighted = tuple(
        item for item in aggregation.seed_estimates if item.smc is not None
    )
    assert len(weighted) == 24
    assert all(item.estimator_applicability == "applicable" for item in weighted)
    assert all(len(item.smc.q0) == 64 for item in weighted)
    assert all(len(item.smc.q1) == 64 for item in weighted)
    assert all(len(item.smc.q2) == 64 for item in weighted)
    assert all(len(item.smc.r1) == 64 for item in weighted)
    assert all(len(item.smc.r2) == 64 for item in weighted)
    assert all(item.smc.y_cross_level_sample_covariances for item in weighted)

    exact = tuple(
        item for item in aggregation.seed_estimates if item.smc is None
    )
    assert len(exact) == 16
    assert all(
        item.estimator_applicability == "not_applicable" for item in exact
    )
    assert all(
        item.applicability_reason
        == "exact_autoregressive_has_no_monte_carlo_estimator"
        for item in exact
    )

    assert result.complete is True
    assert result.objective_status is GateStatus.PASS
    assert result.primary_status is GateStatus.PASS
    assert result.objective_interval is not None
    assert result.primary_interval is not None
    assert result.objective_interval.corner_count == 256
    assert result.primary_interval.corner_count == 256
    assert result.delta == 0.01005033585350145
    assert tuple(row.result_row_key for row in result.result_rows) == (
        inventory.result_row_keys
    )
    assert result.figure_series_keys == inventory.figure_series_keys
    assert result.obligations == ()


def test_ineligible_control_does_not_block_objective_or_primary(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    records = _complete_raw_records(
        inventory,
        tmp_path,
        ineligible_control_arm="WT103-A5-FIXED-COMPLETE-v1",
    )

    aggregation = aggregate_a5_smc(records, inventory=inventory)
    result = paired_prediction_decision(records, inventory=inventory)

    assert aggregation.status is GateStatus.INCONCLUSIVE
    assert result.objective_status is GateStatus.PASS
    assert result.primary_status is GateStatus.PASS
    fixed_row = next(
        row
        for row in result.result_rows
        if row.arm_id == "WT103-A5-FIXED-COMPLETE-v1"
    )
    assert fixed_row.applicability == "descriptive_only"
    assert fixed_row.status is GateStatus.INCONCLUSIVE


def test_objective_and_primary_use_the_frozen_orientations_and_boundaries(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    delta = 0.01005033585350145
    objective_boundary = paired_prediction_decision(
        _complete_raw_records(
            inventory,
            tmp_path / "objective-boundary",
            base_nll_overrides={
                "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1": 1.175,
                "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1": (
                    1.175 + delta
                ),
                "WT103-A0-AR-v1": 1.30,
            },
        ),
        inventory=inventory,
    )
    objective_failure = paired_prediction_decision(
        _complete_raw_records(
            inventory,
            tmp_path / "objective-failure",
            base_nll_overrides={
                "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1": 1.175,
                "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1": (
                    1.175 + delta + 1.0e-5
                ),
                "WT103-A0-AR-v1": 1.30,
            },
        ),
        inventory=inventory,
    )
    primary_below_strict_boundary = paired_prediction_decision(
        _complete_raw_records(
            inventory,
            tmp_path / "primary-boundary",
            base_nll_overrides={
                "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1": 1.18,
                "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1": 1.18,
                "WT103-A0-AR-v1": 1.18 + delta - 1.0e-5,
            },
        ),
        inventory=inventory,
    )
    primary_failure = paired_prediction_decision(
        _complete_raw_records(
            inventory,
            tmp_path / "primary-failure",
            base_nll_overrides={
                "WT103-A5-PARENT-SPECIFIC-PREFIX-EMISSION-v1": 1.18,
                "WT103-A5-PARENT-SPECIFIC-PREFIX-COMPLETE-v1": 1.18,
                "WT103-A0-AR-v1": 1.17,
            },
        ),
        inventory=inventory,
    )

    assert objective_boundary.objective_status is GateStatus.PASS
    assert objective_failure.objective_status is GateStatus.FAIL
    assert objective_failure.primary_status is GateStatus.INCONCLUSIVE
    assert (
        primary_below_strict_boundary.primary_status
        is GateStatus.INCONCLUSIVE
    )
    assert primary_failure.primary_status is GateStatus.FAIL


def test_partial_or_duplicate_raw_inventory_is_inconclusive_and_unaggregated(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    context = _raw_fixture_context(tmp_path, inventory)
    records = _complete_raw_records(
        inventory,
        tmp_path,
        context=context,
    )
    terminal = records[-1]
    protocol = EstimatorProtocol.create()
    spec = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=terminal.particle_count,
        resampling="systematic_ess_half",
    )
    stream = EstimatorStream.create(
        stream_seed=wt103_estimator_stream_seed(
            split="test",
            estimator_protocol_sha256=protocol.protocol_sha256,
            logical_stream_id=terminal.logical_stream_id,
        ),
        estimator_identity=EstimatorIdentity.from_spec(spec),
    )
    binding = WT103EstimatorStreamBinding.create(
        split="test",
        logical_stream_id=terminal.logical_stream_id,
        estimator_protocol=protocol,
        stream=stream,
    )
    failure_payload = b"injected terminal scoring failure"
    failed = WT103RawScoreRecord.create_failed(
        inventory=inventory,
        raw_record_key=terminal.raw_record_key,
        checkpoint=terminal.checkpoint,
        scorer_kind=terminal.scorer_kind,
        logical_stream_id=terminal.logical_stream_id,
        particle_count=terminal.particle_count,
        opening_capability=context.opening_capability,
        evaluation_batches=context.evaluation_batches,
        failure_identity=DurableFileIdentity.create(
            operation="exclusive_create",
            payload=failure_payload,
            volume_identity="task9-failure-volume",
        ),
        failure_reason="injected terminal scoring failure",
        estimator_stream_binding=binding,
        estimator_stream=stream,
    )

    missing = paired_prediction_decision(
        records[:-1],
        inventory=inventory,
    )
    duplicated = paired_prediction_decision(
        records + (records[-1],),
        inventory=inventory,
    )
    failed_result = paired_prediction_decision(
        records[:-1] + (failed,),
        inventory=inventory,
    )

    for result in (missing, duplicated, failed_result):
        assert result.complete is False
        assert result.objective_status is GateStatus.INCONCLUSIVE
        assert result.primary_status is GateStatus.INCONCLUSIVE
        assert result.objective_interval is None
        assert result.primary_interval is None
        assert result.result_rows == ()
        assert result.obligations
    assert failed.disposition == "failed"
    assert failed.evaluation is None
    assert "failed raw test records prevent aggregation" in (
        failed_result.obligations
    )
    with pytest.raises(ValueError, match="enveloped"):
        aggregate_a5_smc(
            (records[0].evaluation,),  # type: ignore[arg-type]
            inventory=inventory,
        )


class _MemoryDurabilityBackend:
    def __init__(
        self,
        *,
        invalid_create_identity: bool = False,
        forged_payload_identity: bool = False,
    ) -> None:
        self.payloads: dict[Path, bytes] = {}
        self.operations: list[str] = []
        self.invalid_create_identity = invalid_create_identity
        self.forged_payload_identity = forged_payload_identity

    def probe(self, root: Path) -> object:
        del root
        return object()

    def create_exclusive(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if path in self.payloads:
            raise DurabilityCollisionError("reservation already exists")
        self.payloads[path] = payload
        self.operations.append("exclusive_create")
        if self.invalid_create_identity:
            return object()  # type: ignore[return-value]
        if self.forged_payload_identity:
            return DurableFileIdentity.create(
                operation="exclusive_create",
                payload=b"x" * len(payload),
                volume_identity="memory-test-volume",
            )
        return DurableFileIdentity.create(
            operation="exclusive_create",
            payload=payload,
            volume_identity="memory-test-volume",
        )

    def replace_durable(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if path not in self.payloads:
            raise RuntimeError("replacement target is absent")
        self.payloads[path] = payload
        self.operations.append("replace")
        return DurableFileIdentity.create(
            operation="replace",
            payload=payload,
            volume_identity="memory-test-volume",
        )

    def publish_bytes(
        self,
        path: Path,
        payload: bytes,
    ) -> DurableFileIdentity:
        if path in self.payloads:
            return self.replace_durable(path, payload)
        return self.create_exclusive(path, payload)


def _opening_plan(
    tmp_path: Path,
    *,
    backend: _MemoryDurabilityBackend,
    checkpoints: tuple[WT103CheckpointIdentity, ...] | None = None,
    run_group_complete: bool = True,
) -> WT103TestOpeningPlan:
    inventory = _inventory()
    exact_checkpoints = (
        tuple(_checkpoint(key) for key in inventory.terminal_checkpoint_keys)
        if checkpoints is None
        else checkpoints
    )
    return WT103TestOpeningPlan.create(
        repository_root=tmp_path,
        durability_backend=backend,
        endpoint_inventory=inventory,
        terminal_checkpoints=exact_checkpoints,
        run_group_complete=run_group_complete,
        run_group_manifest_sha256="1" * 64,
        analysis_sha256="2" * 64,
        figure_sha256="3" * 64,
        data_identity_sha256="4" * 64,
        tokenizer_identity_sha256="5" * 64,
        test_window_manifest_sha256="5" * 64,
        test_schedule_sha256="6" * 64,
    )


def test_test_opening_is_one_immutable_reserved_create_and_reopen_verified(
    tmp_path: Path,
) -> None:
    backend = _MemoryDurabilityBackend()
    plan = _opening_plan(tmp_path, backend=backend)

    capability = reserve_test_opening(plan)

    assert type(capability) is DurableTestOpeningCapability
    assert capability.opening_count == 1
    assert capability.endpoint_inventory_sha256 == (
        plan.endpoint_inventory.endpoint_inventory_sha256
    )
    assert backend.operations == ["exclusive_create"]
    assert b'"state":"RESERVED"' in backend.payloads[plan.reservation_path]
    assert capability.reservation_reopen_verified is True
    with pytest.raises(DurabilityCollisionError):
        reserve_test_opening(plan)


def test_test_opening_rejects_noncanonical_reservation_path(
    tmp_path: Path,
) -> None:
    backend = _MemoryDurabilityBackend()
    plan = _opening_plan(tmp_path, backend=backend)
    alternate = replace(
        plan,
        reservation_path=tmp_path / "second-test-opening-reservation.json",
    )

    with pytest.raises(ValueError, match="canonical reservation path"):
        reserve_test_opening(alternate)

    assert backend.operations == []


def test_test_opening_scope_cannot_be_bypassed_by_new_analysis_or_figure(
    tmp_path: Path,
) -> None:
    backend = _MemoryDurabilityBackend()
    first = _opening_plan(tmp_path, backend=backend)
    second = WT103TestOpeningPlan.create(
        repository_root=tmp_path,
        durability_backend=backend,
        endpoint_inventory=first.endpoint_inventory,
        terminal_checkpoints=first.terminal_checkpoints,
        run_group_complete=first.run_group_complete,
        run_group_manifest_sha256=first.run_group_manifest_sha256,
        analysis_sha256="a" * 64,
        figure_sha256="b" * 64,
        data_identity_sha256=first.data_identity_sha256,
        tokenizer_identity_sha256=first.tokenizer_identity_sha256,
        test_window_manifest_sha256=first.test_window_manifest_sha256,
        test_schedule_sha256=first.test_schedule_sha256,
    )

    assert first.opening_plan_sha256 != second.opening_plan_sha256
    assert first.reservation_scope_sha256 == second.reservation_scope_sha256
    assert first.reservation_path == second.reservation_path
    reserve_test_opening(first)
    with pytest.raises(DurabilityCollisionError):
        reserve_test_opening(second)


def test_test_opening_authenticates_reopened_reservation_bytes(
    tmp_path: Path,
) -> None:
    backend = _MemoryDurabilityBackend(forged_payload_identity=True)
    plan = _opening_plan(tmp_path, backend=backend)

    with pytest.raises(RuntimeError, match="terminal after reservation"):
        reserve_test_opening(plan)

    assert backend.operations == ["exclusive_create"]


def test_test_opening_and_raw_envelope_reject_untyped_data_identity(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    checkpoints = tuple(
        _checkpoint(key) for key in inventory.terminal_checkpoint_keys
    )
    for lane in ("unseal", "raw"):
        lane_root = tmp_path / lane
        lane_root.mkdir()
        windows = _sealed_test_windows(lane_root)
        schedule = build_evaluation_schedule(windows)
        plan = WT103TestOpeningPlan.create(
            repository_root=lane_root,
            durability_backend=_RawOpeningBackend(),
            endpoint_inventory=inventory,
            terminal_checkpoints=checkpoints,
            run_group_complete=True,
            run_group_manifest_sha256="1" * 64,
            analysis_sha256="2" * 64,
            figure_sha256="3" * 64,
            data_identity_sha256="f" * 64,
            tokenizer_identity_sha256=windows.tokenizer_spec.spec_sha256,
            test_window_manifest_sha256=windows.manifest.manifest_sha256,
            test_schedule_sha256=schedule.schedule_sha256,
        )
        capability = reserve_test_opening(plan)
        if lane == "unseal":
            with pytest.raises(ValueError, match="data/window/tokenizer"):
                test_opening_module._unseal_test_windows(  # noqa: SLF001
                    capability,
                    lambda: windows,
                )
            continue
        evaluation_batches = WT103EvaluationBatches.create(
            windows=windows,
            schedule=schedule,
        )
        with pytest.raises(ValueError, match="data/window/tokenizer"):
            test_opening_module._unseal_test_windows(  # noqa: SLF001
                capability,
                lambda: windows,
            )
        with pytest.raises(
            ValueError,
            match="opening capability.*typed test batches",
        ):
            WT103RawScoreRecord.create_finalized(
                inventory=inventory,
                evaluation=None,  # type: ignore[arg-type]
                opening_capability=capability,
                evaluation_batches=evaluation_batches,
                score_trace=None,  # type: ignore[arg-type]
            )


def test_opening_preflight_rejects_incomplete_or_nonterminal_before_create(
    tmp_path: Path,
) -> None:
    backend = _MemoryDurabilityBackend()
    inventory = _inventory()
    complete = tuple(
        _checkpoint(key) for key in inventory.terminal_checkpoint_keys
    )
    wrong_role = (
        _checkpoint(complete[0].logical_key, role="resume_only"),
        *complete[1:],
    )

    with pytest.raises(ValueError, match="terminal_scoring"):
        reserve_test_opening(
            _opening_plan(
                tmp_path,
                backend=backend,
                checkpoints=wrong_role,
            )
        )
    with pytest.raises(ValueError, match="complete run group"):
        reserve_test_opening(
            _opening_plan(
                tmp_path,
                backend=backend,
                run_group_complete=False,
            )
        )
    with pytest.raises(ValueError, match="checkpoint inventory"):
        reserve_test_opening(
            _opening_plan(
                tmp_path,
                backend=backend,
                checkpoints=complete[:-1],
            )
        )

    assert backend.operations == []


def test_opening_active_marker_and_post_reservation_crash_are_terminal(
    tmp_path: Path,
) -> None:
    active_backend = _MemoryDurabilityBackend()
    active_plan = _opening_plan(tmp_path, backend=active_backend)
    marker = tmp_path / ".verification" / "active.json"
    marker.parent.mkdir()
    marker.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="active verification marker"):
        reserve_test_opening(active_plan)
    assert active_backend.operations == []

    marker.unlink()
    failing_backend = _MemoryDurabilityBackend(invalid_create_identity=True)
    failing_plan = _opening_plan(tmp_path, backend=failing_backend)
    with pytest.raises(RuntimeError, match="terminal after reservation"):
        reserve_test_opening(failing_plan)
    assert failing_backend.operations == ["exclusive_create"]
    with pytest.raises(DurabilityCollisionError):
        reserve_test_opening(failing_plan)


def test_opening_plan_hash_is_recomputed_and_unsealer_is_deferred_once(
    tmp_path: Path,
) -> None:
    backend = _MemoryDurabilityBackend()
    plan = _opening_plan(tmp_path, backend=backend)
    with pytest.raises(ValueError, match="opening_plan_sha256"):
        replace(plan, analysis_sha256="9" * 64)

    capability = reserve_test_opening(plan)
    opened = 0

    def fail_after_open_authority() -> object:
        nonlocal opened
        opened += 1
        raise RuntimeError("injected test materialization failure")

    with pytest.raises(RuntimeError, match="materialization"):
        test_opening_module._unseal_test_windows(  # noqa: SLF001
            capability,
            fail_after_open_authority,  # type: ignore[arg-type]
        )
    assert opened == 1
    with pytest.raises(ValueError, match="already been consumed"):
        test_opening_module._unseal_test_windows(  # noqa: SLF001
            capability,
            fail_after_open_authority,  # type: ignore[arg-type]
        )
    assert opened == 1


def test_opening_capability_is_issuer_only() -> None:
    with pytest.raises(TypeError):
        DurableTestOpeningCapability()  # type: ignore[call-arg]


def test_private_test_unsealer_has_one_definition_and_no_import_path() -> None:
    source_root = Path(__file__).resolve().parents[2] / "vfe4"
    occurrences = tuple(
        path
        for path in source_root.rglob("*.py")
        if "_unseal_test_windows" in path.read_text(encoding="utf-8")
    )
    assert occurrences == (
        source_root / "evaluation" / "test_opening.py",
    )
    assert "_unseal_test_windows" not in test_opening_module.__all__
