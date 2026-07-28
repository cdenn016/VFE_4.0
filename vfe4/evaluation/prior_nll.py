"""Target-blind corpus-summed prior-NLL scoring for H6 and WikiText-103."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import torch

from vfe4.data.windows import (
    CausalBatch,
    CausalPrefix,
    CausalWindowSet,
    CausalWindows,
    SEQUENCE_LENGTH,
    WT103_EOT_TOKEN_ID,
    WT103_IGNORE_TARGET_ID,
    WindowSchedule,
    WindowManifest,
    build_evaluation_schedule,
    enumerate_wt103_window_rows,
    iter_causal_batches,
)
from vfe4.predictive import (
    CounterPurpose,
    EstimatorIdentity,
    EstimatorRecord,
    EstimatorStream,
    PrefixCache,
    PriorPrediction,
    PriorPredictor,
    vocabulary_identity_sha256,
)
from vfe4.types import (
    ArmId,
    BoundedPrefixCertificate,
    BoundedPrefixReportReference,
    EstimatorSpec,
    EvidenceStatus,
    H6_PREFIX_REQUIRED_CHECKS,
    NllTotals,
    PrefixCertificate,
    EstimatorProtocol,
    WT103NllTotals,
    VocabularyIdentity,
)
from vfe4.types.training import WT103_PARTICLE_COUNTS, owned_sha256


_PARTICLE_COUNTS = (128, 256, 512, 1024)
_WT103_ESTIMATOR_ROOT_SEED = 2026072198
_WT103_STREAM_SEED_DOMAIN = (
    b"VFE4-WT103-ESTIMATOR-STREAM-SEED-V1\x00"
)
_WT103_STREAM_DOMAINS = {
    "validation": "post-h8-wt103-validation-v1",
    "test": "post-h8-wt103-test-v1",
}


def _validate_scoring_contract(
    *,
    predictor: PriorPredictor,
    windows: CausalWindows,
    estimator_stream: EstimatorStream,
    particle_count: Literal[128, 256, 512, 1024],
    certificate: PrefixCertificate,
) -> VocabularyIdentity:
    if not isinstance(predictor, PriorPredictor):
        raise ValueError("predictor must implement the exact PriorPredictor")
    if type(windows) is not CausalWindows:
        raise ValueError("windows must be an exact CausalWindows")
    if any(
        type(count) is not int or not 1 <= count <= SEQUENCE_LENGTH
        for count in windows.real_target_counts
    ):
        raise ValueError("causal-window horizons must remain in 1..32")
    if type(estimator_stream) is not EstimatorStream:
        raise ValueError("estimator_stream must be an exact EstimatorStream")
    estimator_stream.__post_init__()
    if type(particle_count) is not int or particle_count not in _PARTICLE_COUNTS:
        raise ValueError("particle_count is outside the frozen endpoint ladder")
    if type(certificate) is not PrefixCertificate:
        raise ValueError("certificate must be an exact PrefixCertificate")
    certificate.__post_init__()
    if certificate.status is not EvidenceStatus.PASS:
        raise ValueError("prior scoring requires an exact PASS PrefixCertificate")

    vocabulary = getattr(predictor, "vocabulary", None)
    estimator_spec = getattr(predictor, "estimator_spec", None)
    estimator_identity = getattr(predictor, "estimator_identity", None)
    if type(vocabulary) is not VocabularyIdentity:
        raise ValueError("predictor lacks an exact VocabularyIdentity")
    vocabulary.__post_init__()
    if type(estimator_spec) is not EstimatorSpec:
        raise ValueError("predictor lacks an exact EstimatorSpec")
    estimator_spec.__post_init__()
    if type(estimator_identity) is not EstimatorIdentity:
        raise ValueError("predictor lacks an exact EstimatorIdentity")
    estimator_identity.__post_init__()
    if (
        estimator_spec.kind != "weighted_smc"
        or estimator_spec.particle_count != particle_count
        or getattr(predictor, "particle_count", None) != particle_count
    ):
        raise ValueError("predictor does not match the requested particle count")
    if (
        estimator_identity.semantic_sha256 != estimator_spec.estimator_sha256
        or estimator_stream.estimator_semantic_sha256
        != estimator_identity.semantic_sha256
        or estimator_stream.estimator_artifact_bytes_sha256
        != estimator_identity.artifact_bytes_sha256
        or estimator_stream.estimator_identity_sha256
        != estimator_identity.identity_sha256
    ):
        raise ValueError("estimator stream identity does not match the predictor")
    proposal = getattr(predictor, "proposal", None)
    if (
        getattr(predictor, "vocabulary_sha256", None)
        != vocabulary_identity_sha256(vocabulary)
        or getattr(predictor, "model_family_sha256", None)
        != getattr(proposal, "model_family_sha256", None)
        or getattr(predictor, "model_state_sha256", None)
        != getattr(proposal, "model_state_sha256", None)
        or getattr(predictor, "proposal_identity_sha256", None)
        != getattr(proposal, "proposal_identity_sha256", None)
    ):
        raise ValueError("predictor identities do not match its proposal")

    key = certificate.key
    model = getattr(proposal, "model", None)
    model_arm = getattr(model, "arm", None)
    if model_arm is None:
        model_arm = {
            "a0_causal_transformer": ArmId.A0,
            "a5_mean_pooled_nolatent_floor": ArmId.A5,
        }.get(getattr(model, "family_label", None))
    if type(model_arm) is not ArmId or key.arm is not model_arm:
        raise ValueError("PrefixCertificate arm does not match the predictor")
    expected_certificate_fields = (
        (
            key.predictor_config_sha256,
            getattr(predictor, "predictor_config_sha256", None),
        ),
        (key.estimator_sha256, estimator_spec.estimator_sha256),
        (
            key.model_family_sha256,
            getattr(predictor, "model_family_sha256", None),
        ),
        (
            key.vocabulary_sha256,
            getattr(predictor, "vocabulary_sha256", None),
        ),
        (
            key.data_safety_sha256,
            getattr(predictor, "data_safety_sha256", None),
        ),
    )
    if any(
        observed != expected
        for observed, expected in expected_certificate_fields
    ):
        raise ValueError("PrefixCertificate identities do not match the predictor")
    if proposal is None or not hasattr(proposal, "assert_current_state"):
        raise ValueError("predictor lacks a current-state validation boundary")
    proposal.assert_current_state()
    return vocabulary


def _validate_prediction(
    prediction: object,
    *,
    prefix: CausalPrefix,
    predictor: PriorPredictor,
    estimator_stream: EstimatorStream,
    particle_count: int,
    certificate: PrefixCertificate,
) -> PriorPrediction:
    if type(prediction) is not PriorPrediction:
        raise ValueError("predictor must return an exact PriorPrediction")
    prediction.__post_init__()
    if type(prediction.cache) is not PrefixCache:
        raise ValueError("PriorPrediction must carry an exact PrefixCache")
    prediction.cache.__post_init__()
    if type(prediction.estimator_record) is not EstimatorRecord:
        raise ValueError("PriorPrediction must carry an exact EstimatorRecord")
    prediction.estimator_record.__post_init__()

    key = prediction.cache.key
    prefix_tokens = tuple(int(value) for value in prefix.token_ids.tolist())
    if (
        prediction.vocabulary != getattr(predictor, "vocabulary", None)
        or key.prefix_sha256 != prefix.prefix_sha256
        or key.prefix_tokens != prefix_tokens
        or key.vocabulary_sha256
        != getattr(predictor, "vocabulary_sha256", None)
        or key.predictor_config_sha256
        != getattr(predictor, "predictor_config_sha256", None)
        or key.model_family_sha256
        != getattr(predictor, "model_family_sha256", None)
        or key.model_state_sha256
        != getattr(predictor, "model_state_sha256", None)
        or key.proposal_identity_sha256
        != getattr(predictor, "proposal_identity_sha256", None)
        or key.estimator_semantic_sha256
        != estimator_stream.estimator_semantic_sha256
        or key.estimator_artifact_bytes_sha256
        != estimator_stream.estimator_artifact_bytes_sha256
        or key.estimator_stream_sha256 != estimator_stream.stream_sha256
        or key.data_safety_sha256 != certificate.key.data_safety_sha256
    ):
        raise ValueError("PriorPrediction cache identity does not match the call")
    record = prediction.estimator_record
    if (
        record.estimator_semantic_sha256
        != estimator_stream.estimator_semantic_sha256
        or record.estimator_artifact_bytes_sha256
        != estimator_stream.estimator_artifact_bytes_sha256
        or record.estimator_stream_sha256 != estimator_stream.stream_sha256
        or record.stream_seed != estimator_stream.stream_seed
    ):
        raise ValueError("estimator record identity does not match the call")
    if any(
        consumption.particle_count != particle_count
        for consumption in prediction.cache.counter_consumption
    ):
        raise ValueError("cache counter trace uses the wrong particle count")
    return prediction


def _score_causal_windows(
    *,
    predictor: PriorPredictor,
    windows: CausalWindows,
    estimator_stream: EstimatorStream,
    particle_count: int,
    certificate: PrefixCertificate,
    vocabulary: VocabularyIdentity,
) -> NllTotals:
    negative_log_terms: list[float] = []
    counted_targets = 0
    for window_index in range(len(windows)):
        cache: PrefixCache | None = None
        scored_history: list[int] = []
        real_target_count = windows.real_target_counts[window_index]
        for position in range(real_target_count):
            prefix = CausalPrefix.create(
                receiver_t=len(scored_history) + 1,
                vocabulary=vocabulary,
                token_ids=torch.tensor(
                    scored_history,
                    dtype=torch.int64,
                    device="cpu",
                ),
            )
            raw_prediction = predictor.next_token_log_probs(
                prefix,
                estimator_stream,
                cache,
            )
            prediction = _validate_prediction(
                raw_prediction,
                prefix=prefix,
                predictor=predictor,
                estimator_stream=estimator_stream,
                particle_count=particle_count,
                certificate=certificate,
            )
            cache = prediction.cache

            # The current target is read only after the target-blind call.
            target_id = windows.targets[window_index][position]
            if target_id == -100:
                continue
            if type(target_id) is not int or not 0 <= target_id < vocabulary.size:
                raise ValueError("target token ID falls outside the vocabulary")
            selected = float(prediction.log_probs.value()[target_id].item())
            if not math.isfinite(selected):
                raise ValueError("selected target log probability must be finite")
            negative_log_terms.append(-selected)
            counted_targets += 1
            scored_history.append(target_id)
    if counted_targets != windows.counted_target_total:
        raise ValueError("counted target total does not match causal windows")
    return NllTotals(
        negative_log_likelihood_sum=float(math.fsum(negative_log_terms)),
        counted_targets=counted_targets,
    )


def score_prior_nll_replicate(
    predictor: PriorPredictor,
    windows: CausalWindows,
    estimator_stream: EstimatorStream,
    particle_count: Literal[128, 256, 512, 1024],
    certificate: PrefixCertificate,
) -> NllTotals:
    """Score one exact certificate-bound target-blind estimator replicate."""

    vocabulary = _validate_scoring_contract(
        predictor=predictor,
        windows=windows,
        estimator_stream=estimator_stream,
        particle_count=particle_count,
        certificate=certificate,
    )
    return _score_causal_windows(
        predictor=predictor,
        windows=windows,
        estimator_stream=estimator_stream,
        particle_count=particle_count,
        certificate=certificate,
        vocabulary=vocabulary,
    )


def _bounded_validation_reference(
    *,
    certificate: BoundedPrefixCertificate,
    particle_count: Literal[128, 256, 512, 1024],
) -> BoundedPrefixReportReference:
    if type(certificate) is not BoundedPrefixCertificate:
        raise ValueError(
            "bounded prior scoring requires an exact BoundedPrefixCertificate"
        )
    certificate.__post_init__()
    if (
        certificate.status is not EvidenceStatus.PASS
        or certificate.obligations != ()
        or tuple(certificate.checks) != H6_PREFIX_REQUIRED_CHECKS
        or any(value is not True for value in certificate.checks.values())
    ):
        raise ValueError(
            "bounded prior scoring requires exact PASS with every Prefix check"
        )
    matching = tuple(
        reference
        for reference in certificate.report_binding.report_references
        if (
            reference.particle_count == particle_count
            and reference.case_family == "validation"
        )
    )
    if len(matching) != 1:
        raise ValueError(
            "bounded Prefix certificate lacks one unique validation report"
        )
    reference = matching[0]
    reference.__post_init__()
    return reference


def _validate_bounded_scoring_contract_v3(
    *,
    predictor: PriorPredictor,
    windows: CausalWindows,
    estimator_stream: EstimatorStream,
    particle_count: Literal[128, 256, 512, 1024],
    certificate: BoundedPrefixCertificate,
) -> tuple[VocabularyIdentity, BoundedPrefixReportReference]:
    if not isinstance(predictor, PriorPredictor):
        raise ValueError("predictor must implement the exact PriorPredictor")
    if type(windows) is not CausalWindows:
        raise ValueError("windows must be an exact CausalWindows")
    if any(
        type(count) is not int or not 1 <= count <= SEQUENCE_LENGTH
        for count in windows.real_target_counts
    ):
        raise ValueError("causal-window horizons must remain in 1..32")
    if type(estimator_stream) is not EstimatorStream:
        raise ValueError("estimator_stream must be an exact EstimatorStream")
    estimator_stream.__post_init__()
    if type(particle_count) is not int or particle_count not in _PARTICLE_COUNTS:
        raise ValueError("particle_count is outside the frozen endpoint ladder")
    reference = _bounded_validation_reference(
        certificate=certificate,
        particle_count=particle_count,
    )

    vocabulary = getattr(predictor, "vocabulary", None)
    estimator_spec = getattr(predictor, "estimator_spec", None)
    estimator_identity = getattr(predictor, "estimator_identity", None)
    if type(vocabulary) is not VocabularyIdentity:
        raise ValueError("predictor lacks an exact VocabularyIdentity")
    vocabulary.__post_init__()
    if type(estimator_spec) is not EstimatorSpec:
        raise ValueError("predictor lacks an exact EstimatorSpec")
    estimator_spec.__post_init__()
    if type(estimator_identity) is not EstimatorIdentity:
        raise ValueError("predictor lacks an exact EstimatorIdentity")
    estimator_identity.__post_init__()
    if (
        estimator_spec.kind != "weighted_smc"
        or estimator_spec.particle_count != particle_count
        or getattr(predictor, "particle_count", None) != particle_count
    ):
        raise ValueError("predictor does not match the requested particle count")
    if (
        estimator_identity.semantic_sha256 != estimator_spec.estimator_sha256
        or estimator_stream.estimator_semantic_sha256
        != estimator_identity.semantic_sha256
        or estimator_stream.estimator_artifact_bytes_sha256
        != estimator_identity.artifact_bytes_sha256
        or estimator_stream.estimator_identity_sha256
        != estimator_identity.identity_sha256
        or reference.estimator_semantic_sha256
        != estimator_identity.semantic_sha256
        or reference.estimator_artifact_bytes_sha256
        != estimator_identity.artifact_bytes_sha256
    ):
        raise ValueError(
            "bounded Prefix estimator identity does not match the predictor"
        )
    proposal = getattr(predictor, "proposal", None)
    if (
        getattr(predictor, "vocabulary_sha256", None)
        != vocabulary_identity_sha256(vocabulary)
        or getattr(predictor, "model_family_sha256", None)
        != getattr(proposal, "model_family_sha256", None)
        or getattr(predictor, "model_state_sha256", None)
        != getattr(proposal, "model_state_sha256", None)
        or getattr(predictor, "proposal_identity_sha256", None)
        != getattr(proposal, "proposal_identity_sha256", None)
    ):
        raise ValueError("predictor identities do not match its proposal")

    key = reference.report_key
    model = getattr(proposal, "model", None)
    model_arm = getattr(model, "arm", None)
    if model_arm is None:
        model_arm = {
            "a0_causal_transformer": ArmId.A0,
            "a5_mean_pooled_nolatent_floor": ArmId.A5,
        }.get(getattr(model, "family_label", None))
    if type(model_arm) is not ArmId or key.arm is not model_arm:
        raise ValueError("bounded Prefix report arm does not match the predictor")
    expected_fields = (
        (
            key.predictor_config_sha256,
            getattr(predictor, "predictor_config_sha256", None),
        ),
        (key.estimator_sha256, estimator_spec.estimator_sha256),
        (
            key.model_family_sha256,
            getattr(predictor, "model_family_sha256", None),
        ),
        (
            key.vocabulary_sha256,
            getattr(predictor, "vocabulary_sha256", None),
        ),
        (
            key.data_safety_sha256,
            getattr(predictor, "data_safety_sha256", None),
        ),
    )
    if any(observed != expected for observed, expected in expected_fields):
        raise ValueError(
            "bounded Prefix report identities do not match the predictor"
        )
    if proposal is None or not hasattr(proposal, "assert_current_state"):
        raise ValueError("predictor lacks a current-state validation boundary")
    proposal.assert_current_state()
    return vocabulary, reference


def _validate_bounded_prediction_v3(
    prediction: object,
    *,
    prefix: CausalPrefix,
    predictor: PriorPredictor,
    estimator_stream: EstimatorStream,
    particle_count: int,
    reference: BoundedPrefixReportReference,
) -> PriorPrediction:
    if type(prediction) is not PriorPrediction:
        raise ValueError("predictor must return an exact PriorPrediction")
    prediction.__post_init__()
    if type(prediction.cache) is not PrefixCache:
        raise ValueError("PriorPrediction must carry an exact PrefixCache")
    prediction.cache.__post_init__()
    if type(prediction.estimator_record) is not EstimatorRecord:
        raise ValueError("PriorPrediction must carry an exact EstimatorRecord")
    prediction.estimator_record.__post_init__()

    key = prediction.cache.key
    prefix_tokens = tuple(int(value) for value in prefix.token_ids.tolist())
    if (
        prediction.vocabulary != getattr(predictor, "vocabulary", None)
        or key.prefix_sha256 != prefix.prefix_sha256
        or key.prefix_tokens != prefix_tokens
        or key.vocabulary_sha256
        != getattr(predictor, "vocabulary_sha256", None)
        or key.predictor_config_sha256
        != getattr(predictor, "predictor_config_sha256", None)
        or key.model_family_sha256
        != getattr(predictor, "model_family_sha256", None)
        or key.model_state_sha256
        != getattr(predictor, "model_state_sha256", None)
        or key.proposal_identity_sha256
        != getattr(predictor, "proposal_identity_sha256", None)
        or key.estimator_semantic_sha256
        != estimator_stream.estimator_semantic_sha256
        or key.estimator_artifact_bytes_sha256
        != estimator_stream.estimator_artifact_bytes_sha256
        or key.estimator_stream_sha256 != estimator_stream.stream_sha256
        or key.data_safety_sha256
        != reference.report_key.data_safety_sha256
    ):
        raise ValueError("PriorPrediction cache identity does not match the call")
    record = prediction.estimator_record
    if (
        record.estimator_semantic_sha256
        != estimator_stream.estimator_semantic_sha256
        or record.estimator_artifact_bytes_sha256
        != estimator_stream.estimator_artifact_bytes_sha256
        or record.estimator_stream_sha256 != estimator_stream.stream_sha256
        or record.stream_seed != estimator_stream.stream_seed
    ):
        raise ValueError("estimator record identity does not match the call")
    if any(
        consumption.particle_count != particle_count
        for consumption in prediction.cache.counter_consumption
    ):
        raise ValueError("cache counter trace uses the wrong particle count")
    return prediction


def _score_bounded_causal_windows_v3(
    *,
    predictor: PriorPredictor,
    windows: CausalWindows,
    estimator_stream: EstimatorStream,
    particle_count: int,
    reference: BoundedPrefixReportReference,
    vocabulary: VocabularyIdentity,
) -> NllTotals:
    negative_log_terms: list[float] = []
    counted_targets = 0
    for window_index in range(len(windows)):
        cache: PrefixCache | None = None
        scored_history: list[int] = []
        real_target_count = windows.real_target_counts[window_index]
        for position in range(real_target_count):
            prefix = CausalPrefix.create(
                receiver_t=len(scored_history) + 1,
                vocabulary=vocabulary,
                token_ids=torch.tensor(
                    scored_history,
                    dtype=torch.int64,
                    device="cpu",
                ),
            )
            raw_prediction = predictor.next_token_log_probs(
                prefix,
                estimator_stream,
                cache,
            )
            prediction = _validate_bounded_prediction_v3(
                raw_prediction,
                prefix=prefix,
                predictor=predictor,
                estimator_stream=estimator_stream,
                particle_count=particle_count,
                reference=reference,
            )
            cache = prediction.cache

            # The current target is intentionally read after the target-blind call.
            target_id = windows.targets[window_index][position]
            if target_id == -100:
                continue
            if type(target_id) is not int or not 0 <= target_id < vocabulary.size:
                raise ValueError("target token ID falls outside the vocabulary")
            selected = float(prediction.log_probs.value()[target_id].item())
            if not math.isfinite(selected):
                raise ValueError("selected target log probability must be finite")
            negative_log_terms.append(-selected)
            counted_targets += 1
            scored_history.append(target_id)
    if counted_targets != windows.counted_target_total:
        raise ValueError("counted target total does not match causal windows")
    return NllTotals(
        negative_log_likelihood_sum=float(math.fsum(negative_log_terms)),
        counted_targets=counted_targets,
    )


def score_bounded_prior_nll_replicate_v3(
    predictor: PriorPredictor,
    windows: CausalWindows,
    estimator_stream: EstimatorStream,
    particle_count: Literal[128, 256, 512, 1024],
    certificate: BoundedPrefixCertificate,
) -> NllTotals:
    """Score one target-blind replicate under an exact bounded v2 certificate."""

    vocabulary, reference = _validate_bounded_scoring_contract_v3(
        predictor=predictor,
        windows=windows,
        estimator_stream=estimator_stream,
        particle_count=particle_count,
        certificate=certificate,
    )
    return _score_bounded_causal_windows_v3(
        predictor=predictor,
        windows=windows,
        estimator_stream=estimator_stream,
        particle_count=particle_count,
        reference=reference,
        vocabulary=vocabulary,
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _wt103_batch_records(
    batches: tuple[CausalBatch, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "window_ids": batch.window_ids,
            "inputs_sha256": _tensor_sha256(batch.inputs),
            "targets_sha256": _tensor_sha256(batch.targets),
            "attention_mask_sha256": _tensor_sha256(
                batch.attention_mask
            ),
            "counted_targets": batch.counted_targets,
        }
        for batch in batches
    )


def _same_wt103_batches(
    left: tuple[CausalBatch, ...],
    right: tuple[CausalBatch, ...],
) -> bool:
    return len(left) == len(right) and all(
        observed.window_ids == expected.window_ids
        and observed.counted_targets == expected.counted_targets
        and torch.equal(observed.inputs, expected.inputs)
        and torch.equal(observed.targets, expected.targets)
        and torch.equal(observed.attention_mask, expected.attention_mask)
        for observed, expected in zip(left, right, strict=True)
    )


def _window_source_identity(
    windows: CausalWindowSet,
    schedule: WindowSchedule,
) -> str:
    rows = enumerate_wt103_window_rows(
        windows.manifest.counted_targets + 1
    )
    if windows.rows != rows:
        raise ValueError("evaluation rows differ from the canonical window source")
    row_payload = b"VFE4-WT103-WINDOW-ROWS-V1\x00" + b"".join(
        row.canonical_bytes() for row in rows
    )
    if hashlib.sha256(row_payload).hexdigest() != windows.manifest.payload_sha256:
        raise ValueError(
            "evaluation row bytes differ from the window source manifest"
        )
    token_hasher = hashlib.sha256()
    token_bytes = memoryview(windows._tokens).cast("B")  # noqa: SLF001
    chunk_size = 8 * 1024 * 1024
    for offset in range(0, len(token_bytes), chunk_size):
        token_hasher.update(token_bytes[offset : offset + chunk_size])
    if token_hasher.hexdigest() != windows.manifest.token_payload_sha256:
        raise ValueError(
            "evaluation token bytes differ from the window source manifest"
        )
    return owned_sha256(
        "vfe4.wt103.evaluation-window-source.v1",
        {
            "cache_record_sha256": windows.cache_record.record_sha256,
            "tokenizer_spec_sha256": windows.tokenizer_spec.spec_sha256,
            "window_manifest_sha256": windows.manifest.manifest_sha256,
            "row_payload_sha256": windows.manifest.payload_sha256,
            "token_payload_sha256": windows.manifest.token_payload_sha256,
            "schedule_sha256": schedule.schedule_sha256,
        },
    )


@dataclass(frozen=True, slots=True)
class WT103EvaluationBatches:
    """One typed-window-bound, complete, ascending evaluation inventory."""

    windows: CausalWindowSet
    schedule: WindowSchedule
    manifest: WindowManifest
    batches: tuple[CausalBatch, ...]
    window_source_sha256: str
    batches_sha256: str

    def __post_init__(self) -> None:
        if type(self.windows) is not CausalWindowSet:
            raise ValueError(
                "evaluation batches require an exact CausalWindowSet source"
            )
        self.windows.__post_init__()
        if type(self.schedule) is not WindowSchedule:
            raise ValueError(
                "evaluation batches require an exact WindowSchedule source"
            )
        self.schedule.__post_init__()
        if type(self.manifest) is not WindowManifest:
            raise ValueError("evaluation batches require an exact WindowManifest")
        self.manifest.__post_init__()
        if self.manifest.split not in ("validation", "test"):
            raise ValueError("prior evaluation requires validation or test data")
        expected_schedule = build_evaluation_schedule(self.windows)
        if (
            self.manifest != self.windows.manifest
            or self.schedule != expected_schedule
            or self.schedule.window_manifest_sha256
            != self.manifest.manifest_sha256
        ):
            raise ValueError(
                "evaluation schedule/manifest differs from the window source"
            )
        expected_source = _window_source_identity(
            self.windows,
            self.schedule,
        )
        if self.window_source_sha256 != expected_source:
            raise ValueError(
                "window_source_sha256 does not match the window source"
            )
        if (
            type(self.batches) is not tuple
            or not self.batches
            or any(type(batch) is not CausalBatch for batch in self.batches)
        ):
            raise ValueError("evaluation batches must be a nonempty exact tuple")
        flattened_ids: list[int] = []
        counted_targets = 0
        for batch in self.batches:
            batch.__post_init__()
            _validate_wt103_batch_rows(batch)
            flattened_ids.extend(batch.window_ids)
            counted_targets += batch.counted_targets
        if tuple(flattened_ids) != tuple(range(self.manifest.window_count)):
            raise ValueError(
                "evaluation batches differ from the complete manifest inventory"
            )
        if counted_targets != self.manifest.counted_targets:
            raise ValueError(
                "evaluation batches differ from the manifest target denominator"
            )
        expected_batches = tuple(
            iter_causal_batches(
                windows=self.windows,
                schedule=self.schedule,
            )
        )
        if not _same_wt103_batches(self.batches, expected_batches):
            raise ValueError(
                "evaluation batches differ from the typed window source"
            )
        expected = owned_sha256(
            "vfe4.wt103.evaluation-batches.v1",
            {
                "window_manifest_sha256": self.manifest.manifest_sha256,
                "schedule_sha256": self.schedule.schedule_sha256,
                "window_source_sha256": self.window_source_sha256,
                "batch_records": _wt103_batch_records(self.batches),
            },
        )
        if self.batches_sha256 != expected:
            raise ValueError("batches_sha256 does not match evaluation batches")

    def __iter__(self):
        return iter(self.batches)

    @classmethod
    def create(
        cls,
        *,
        windows: CausalWindowSet,
        schedule: WindowSchedule,
    ) -> "WT103EvaluationBatches":
        if type(windows) is not CausalWindowSet:
            raise ValueError(
                "evaluation batches require an exact CausalWindowSet source"
            )
        windows.__post_init__()
        if type(schedule) is not WindowSchedule:
            raise ValueError(
                "evaluation batches require an exact WindowSchedule source"
            )
        schedule.__post_init__()
        batches = tuple(
            iter_causal_batches(
                windows=windows,
                schedule=schedule,
            )
        )
        source_sha256 = _window_source_identity(windows, schedule)
        payload = {
            "window_manifest_sha256": windows.manifest.manifest_sha256,
            "schedule_sha256": schedule.schedule_sha256,
            "window_source_sha256": source_sha256,
            "batch_records": _wt103_batch_records(batches),
        }
        return cls(
            windows=windows,
            schedule=schedule,
            manifest=windows.manifest,
            batches=batches,
            window_source_sha256=source_sha256,
            batches_sha256=owned_sha256(
                "vfe4.wt103.evaluation-batches.v1",
                payload,
            ),
        )


def wt103_common_stream_registry_sha256(
    *,
    split: Literal["validation", "test"],
    estimator_protocol_sha256: str,
    logical_stream_id: int | None,
) -> str:
    """Return the frozen purpose registry for one logical WT103 stream."""

    if split not in ("validation", "test"):
        raise ValueError("WT103 stream registry split is invalid")
    if logical_stream_id is not None and (
        type(logical_stream_id) is not int or logical_stream_id < 0
    ):
        raise ValueError("logical stream ID must be nonnegative or absent")
    if (
        type(estimator_protocol_sha256) is not str
        or len(estimator_protocol_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in estimator_protocol_sha256
        )
    ):
        raise ValueError("estimator_protocol_sha256 must be a SHA-256 identity")
    logical_label: int | str = (
        "exact" if logical_stream_id is None else logical_stream_id
    )
    domain = _WT103_STREAM_DOMAINS[split]
    return owned_sha256(
        f"vfe4.wt103.common-{split}-stream.v1",
        {
            "estimator_protocol_sha256": estimator_protocol_sha256,
            "stream_id": logical_stream_id,
            "purpose_stream_sha256s": tuple(
                (
                    purpose.value,
                    hashlib.sha256(
                        (
                            f"{domain}|{_WT103_ESTIMATOR_ROOT_SEED}|"
                            f"{logical_label}|{purpose.value}"
                        ).encode("ascii")
                    ).hexdigest(),
                )
                for purpose in CounterPurpose
            ),
        },
    )


def wt103_estimator_stream_seed(
    *,
    split: Literal["validation", "test"],
    estimator_protocol_sha256: str,
    logical_stream_id: int | None,
) -> int:
    """Derive the sole unsigned-64 counter seed for a logical stream."""

    registry = wt103_common_stream_registry_sha256(
        split=split,
        estimator_protocol_sha256=estimator_protocol_sha256,
        logical_stream_id=logical_stream_id,
    )
    split_bytes = split.encode("ascii")
    logical_bytes = (
        b"exact"
        if logical_stream_id is None
        else logical_stream_id.to_bytes(8, "little", signed=False)
    )
    digest = hashlib.sha256(
        _WT103_STREAM_SEED_DOMAIN
        + len(split_bytes).to_bytes(1, "little")
        + split_bytes
        + _WT103_ESTIMATOR_ROOT_SEED.to_bytes(
            8,
            "little",
            signed=False,
        )
        + len(logical_bytes).to_bytes(1, "little")
        + logical_bytes
        + bytes.fromhex(registry)
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


@dataclass(frozen=True, slots=True)
class WT103EstimatorStreamBinding:
    """Bind one concrete estimator stream to one preregistered logical ID."""

    split: Literal["validation", "test"]
    logical_stream_id: int | None
    estimator_protocol_sha256: str
    estimator_stream_sha256: str
    stream_seed: int
    common_stream_registry_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        if self.split not in ("validation", "test"):
            raise ValueError("estimator stream binding split is invalid")
        if self.logical_stream_id is not None and (
            type(self.logical_stream_id) is not int
            or self.logical_stream_id < 0
            or (
                self.split == "validation"
                and self.logical_stream_id >= 8
            )
            or (
                self.split == "test"
                and self.logical_stream_id >= 64
            )
        ):
            raise ValueError(
                "logical stream ID is outside the frozen split inventory"
            )
        payload = {
            "split": self.split,
            "logical_stream_id": self.logical_stream_id,
            "estimator_protocol_sha256": self.estimator_protocol_sha256,
            "estimator_stream_sha256": self.estimator_stream_sha256,
            "stream_seed": self.stream_seed,
            "common_stream_registry_sha256": (
                self.common_stream_registry_sha256
            ),
        }
        for name in (
            "estimator_protocol_sha256",
            "estimator_stream_sha256",
            "common_stream_registry_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 identity")
        if (
            self.estimator_protocol_sha256
            != EstimatorProtocol.create().protocol_sha256
        ):
            raise ValueError(
                "stream binding differs from the frozen estimator protocol"
            )
        expected_registry = wt103_common_stream_registry_sha256(
            split=self.split,
            estimator_protocol_sha256=self.estimator_protocol_sha256,
            logical_stream_id=self.logical_stream_id,
        )
        expected_seed = wt103_estimator_stream_seed(
            split=self.split,
            estimator_protocol_sha256=self.estimator_protocol_sha256,
            logical_stream_id=self.logical_stream_id,
        )
        if (
            self.common_stream_registry_sha256 != expected_registry
            or self.stream_seed != expected_seed
        ):
            raise ValueError(
                "stream binding does not use the canonical counter seed"
            )
        expected = owned_sha256(
            "vfe4.wt103.estimator-stream-binding.v1",
            payload,
        )
        if self.binding_sha256 != expected:
            raise ValueError("binding_sha256 does not match stream binding")

    @classmethod
    def create(
        cls,
        *,
        split: Literal["validation", "test"],
        logical_stream_id: int | None,
        estimator_protocol: EstimatorProtocol,
        stream: EstimatorStream,
    ) -> "WT103EstimatorStreamBinding":
        if type(estimator_protocol) is not EstimatorProtocol:
            raise ValueError("binding requires an exact EstimatorProtocol")
        estimator_protocol.__post_init__()
        if type(stream) is not EstimatorStream:
            raise ValueError("binding requires an exact EstimatorStream")
        stream.__post_init__()
        allowed_ids = (
            estimator_protocol.validation_stream_ids
            if split == "validation"
            else estimator_protocol.test_stream_ids
        )
        if logical_stream_id is not None and logical_stream_id not in allowed_ids:
            raise ValueError("logical stream ID is outside the frozen protocol")
        expected_seed = wt103_estimator_stream_seed(
            split=split,
            estimator_protocol_sha256=estimator_protocol.protocol_sha256,
            logical_stream_id=logical_stream_id,
        )
        if stream.stream_seed != expected_seed:
            raise ValueError(
                "stream does not use the canonical counter seed for its "
                "logical ID"
            )
        payload = {
            "split": split,
            "logical_stream_id": logical_stream_id,
            "estimator_protocol_sha256": estimator_protocol.protocol_sha256,
            "estimator_stream_sha256": stream.stream_sha256,
            "stream_seed": stream.stream_seed,
            "common_stream_registry_sha256": (
                wt103_common_stream_registry_sha256(
                    split=split,
                    estimator_protocol_sha256=(
                        estimator_protocol.protocol_sha256
                    ),
                    logical_stream_id=logical_stream_id,
                )
            ),
        }
        return cls(
            **payload,
            binding_sha256=owned_sha256(
                "vfe4.wt103.estimator-stream-binding.v1",
                payload,
            ),
        )


@dataclass(frozen=True, slots=True)
class WT103ScoreTrace:
    """Typed batch/stream/counter evidence for one corpus score."""

    schema_version: Literal["wt103-score-trace-v1"]
    evaluation_batches: WT103EvaluationBatches
    binding: WT103EstimatorStreamBinding
    stream: EstimatorStream
    totals: WT103NllTotals
    estimator_records: tuple[EstimatorRecord, ...]
    negative_log_terms: tuple[float, ...]
    counter_trace_sha256: str | None
    counter_draw_count: int | None
    trace_sha256: str

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "batches_sha256": self.evaluation_batches.batches_sha256,
            "window_source_sha256": (
                self.evaluation_batches.window_source_sha256
            ),
            "schedule_sha256": (
                self.evaluation_batches.schedule.schedule_sha256
            ),
            "binding_sha256": self.binding.binding_sha256,
            "estimator_stream_sha256": self.stream.stream_sha256,
            "stream_seed": self.stream.stream_seed,
            "totals_sha256": self.totals.totals_sha256,
            "estimator_record_sha256s": tuple(
                record.record_sha256 for record in self.estimator_records
            ),
            "negative_log_terms": self.negative_log_terms,
            "counter_trace_sha256": self.counter_trace_sha256,
            "counter_draw_count": self.counter_draw_count,
        }

    def __post_init__(self) -> None:
        if self.schema_version != "wt103-score-trace-v1":
            raise ValueError("WT103 score trace schema is invalid")
        if type(self.evaluation_batches) is not WT103EvaluationBatches:
            raise ValueError(
                "WT103 score trace requires exact evaluation batches"
            )
        self.evaluation_batches.__post_init__()
        if type(self.binding) is not WT103EstimatorStreamBinding:
            raise ValueError("WT103 score trace requires an exact binding")
        self.binding.__post_init__()
        if type(self.stream) is not EstimatorStream:
            raise ValueError("WT103 score trace requires an exact stream")
        self.stream.__post_init__()
        if type(self.totals) is not WT103NllTotals:
            raise ValueError("WT103 score trace requires exact NLL totals")
        self.totals.__post_init__()
        if (
            type(self.estimator_records) is not tuple
            or len(self.estimator_records) != self.totals.counted_targets
            or any(
                type(record) is not EstimatorRecord
                for record in self.estimator_records
            )
        ):
            raise ValueError(
                "WT103 score trace requires one typed estimator record "
                "per counted target"
            )
        if (
            type(self.negative_log_terms) is not tuple
            or len(self.negative_log_terms) != self.totals.counted_targets
            or any(
                type(term) is not float
                or not math.isfinite(term)
                or term < 0.0
                for term in self.negative_log_terms
            )
            or math.fsum(self.negative_log_terms)
            != self.totals.summed_nll
            or self.totals.counted_targets
            != self.evaluation_batches.manifest.counted_targets
        ):
            raise ValueError(
                "WT103 score trace terms do not reproduce corpus totals"
            )
        for record in self.estimator_records:
            record.__post_init__()
            if (
                record.estimator_stream_sha256
                != self.stream.stream_sha256
                or record.stream_seed != self.stream.stream_seed
                or record.estimator_semantic_sha256
                != self.stream.estimator_semantic_sha256
                or record.estimator_artifact_bytes_sha256
                != self.stream.estimator_artifact_bytes_sha256
            ):
                raise ValueError(
                    "estimator record differs from the bound score stream"
                )
        if (
            self.binding.estimator_stream_sha256
            != self.stream.stream_sha256
            or self.binding.stream_seed != self.stream.stream_seed
            or self.binding.split != self.evaluation_batches.manifest.split
            or self.totals.estimator_stream_id
            != self.binding.logical_stream_id
        ):
            raise ValueError("WT103 score trace cross-links disagree")
        observed_draws = sum(
            record.counter_draw_count for record in self.estimator_records
        )
        if self.totals.scorer_kind == "exact_autoregressive":
            if (
                self.binding.logical_stream_id is not None
                or self.totals.particle_count is not None
                or self.counter_trace_sha256 is not None
                or self.counter_draw_count is not None
                or observed_draws != 0
            ):
                raise ValueError(
                    "exact score trace cannot fabricate counter evidence"
                )
        else:
            expected_counter = owned_sha256(
                "vfe4.wt103.score-counter-trace.v1",
                tuple(
                    (
                        record.prefix_sha256,
                        record.counter_trace_sha256,
                        record.counter_draw_count,
                    )
                    for record in self.estimator_records
                ),
            )
            if (
                self.binding.logical_stream_id is None
                or self.totals.particle_count not in WT103_PARTICLE_COUNTS
                or self.counter_trace_sha256 != expected_counter
                or self.counter_draw_count != observed_draws
                or observed_draws <= 0
            ):
                raise ValueError(
                    "weighted score trace counter evidence is not canonical"
                )
        expected = owned_sha256(
            "vfe4.wt103.score-trace.v1",
            self._payload(),
        )
        if self.trace_sha256 != expected:
            raise ValueError("trace_sha256 does not match the score trace")

    @classmethod
    def create(
        cls,
        *,
        evaluation_batches: WT103EvaluationBatches,
        binding: WT103EstimatorStreamBinding,
        stream: EstimatorStream,
        totals: WT103NllTotals,
        estimator_records: tuple[EstimatorRecord, ...],
        negative_log_terms: tuple[float, ...],
    ) -> "WT103ScoreTrace":
        counter_draw_count: int | None
        counter_trace_sha256: str | None
        if totals.scorer_kind == "exact_autoregressive":
            counter_draw_count = None
            counter_trace_sha256 = None
        else:
            counter_draw_count = sum(
                record.counter_draw_count for record in estimator_records
            )
            counter_trace_sha256 = owned_sha256(
                "vfe4.wt103.score-counter-trace.v1",
                tuple(
                    (
                        record.prefix_sha256,
                        record.counter_trace_sha256,
                        record.counter_draw_count,
                    )
                    for record in estimator_records
                ),
            )
        payload = {
            "schema_version": "wt103-score-trace-v1",
            "batches_sha256": evaluation_batches.batches_sha256,
            "window_source_sha256": (
                evaluation_batches.window_source_sha256
            ),
            "schedule_sha256": evaluation_batches.schedule.schedule_sha256,
            "binding_sha256": binding.binding_sha256,
            "estimator_stream_sha256": stream.stream_sha256,
            "stream_seed": stream.stream_seed,
            "totals_sha256": totals.totals_sha256,
            "estimator_record_sha256s": tuple(
                record.record_sha256 for record in estimator_records
            ),
            "negative_log_terms": negative_log_terms,
            "counter_trace_sha256": counter_trace_sha256,
            "counter_draw_count": counter_draw_count,
        }
        return cls(
            schema_version="wt103-score-trace-v1",
            evaluation_batches=evaluation_batches,
            binding=binding,
            stream=stream,
            totals=totals,
            estimator_records=estimator_records,
            negative_log_terms=negative_log_terms,
            counter_trace_sha256=counter_trace_sha256,
            counter_draw_count=counter_draw_count,
            trace_sha256=owned_sha256(
                "vfe4.wt103.score-trace.v1",
                payload,
            ),
        )


@dataclass(slots=True)
class _IssuedScoreTrace:
    totals: WT103NllTotals
    trace: WT103ScoreTrace


_ISSUED_SCORE_TRACES: dict[int, _IssuedScoreTrace] = {}


def wt103_score_trace(totals: WT103NllTotals) -> WT103ScoreTrace:
    """Return the typed trace issued with a live ``score_prior_nll`` result."""

    issued = _ISSUED_SCORE_TRACES.get(id(totals))
    if issued is None or issued.totals is not totals:
        raise ValueError("NLL totals have no issued WT103 score trace")
    issued.trace.__post_init__()
    return issued.trace


class _BoundWT103PriorPredictor:
    def __init__(
        self,
        predictor: PriorPredictor,
        binding: WT103EstimatorStreamBinding,
    ) -> None:
        self._predictor = predictor
        self.wt103_stream_binding = binding
        self.vocabulary = getattr(predictor, "vocabulary", None)
        self.estimator_spec = getattr(predictor, "estimator_spec", None)
        self.estimator_identity = getattr(predictor, "estimator_identity", None)

    def next_token_log_probs(
        self,
        prefix_tokens: CausalPrefix,
        estimator_rng: EstimatorStream,
        cache: PrefixCache | None = None,
    ) -> PriorPrediction:
        return self._predictor.next_token_log_probs(
            prefix_tokens,
            estimator_rng,
            cache,
        )


def bind_wt103_prior_predictor(
    predictor: PriorPredictor,
    binding: WT103EstimatorStreamBinding,
) -> PriorPredictor:
    """Bind a predictor to a logical stream without changing its protocol."""

    if not isinstance(predictor, PriorPredictor):
        raise ValueError("predictor must implement PriorPredictor")
    if type(binding) is not WT103EstimatorStreamBinding:
        raise ValueError("binding must be an exact WT103 stream binding")
    binding.__post_init__()
    return _BoundWT103PriorPredictor(predictor, binding)


def _validate_wt103_batch_rows(batch: CausalBatch) -> None:
    for row_index in range(len(batch.window_ids)):
        mask = batch.attention_mask[row_index]
        real_count = int(torch.sum(mask).item())
        if (
            real_count <= 0
            or not bool(torch.all(mask[:real_count]))
            or bool(torch.any(mask[real_count:]))
            or not bool(
                torch.all(
                    batch.targets[row_index, real_count:]
                    == WT103_IGNORE_TARGET_ID
                )
            )
            or not bool(
                torch.all(
                    batch.inputs[row_index, real_count:]
                    == WT103_EOT_TOKEN_ID
                )
            )
        ):
            raise ValueError("WT103 evaluation row padding/mask is not canonical")
        if real_count > 1 and not torch.equal(
            batch.inputs[row_index, 1:real_count],
            batch.targets[row_index, : real_count - 1],
        ):
            raise ValueError("WT103 evaluation row is not shifted causal data")


def _validate_wt103_scoring_inputs(
    *,
    predictor: PriorPredictor,
    batches: Iterable[CausalBatch],
    stream: EstimatorStream,
) -> tuple[
    WT103EvaluationBatches,
    VocabularyIdentity,
    EstimatorSpec,
    WT103EstimatorStreamBinding,
]:
    if type(predictor) is not _BoundWT103PriorPredictor:
        raise ValueError(
            "predictor must be bound to a WT103 logical estimator stream"
        )
    if type(stream) is not EstimatorStream:
        raise ValueError("stream must be an exact EstimatorStream")
    stream.__post_init__()
    vocabulary = getattr(predictor, "vocabulary", None)
    estimator_spec = getattr(predictor, "estimator_spec", None)
    estimator_identity = getattr(predictor, "estimator_identity", None)
    binding = predictor.wt103_stream_binding
    binding.__post_init__()
    if type(vocabulary) is not VocabularyIdentity:
        raise ValueError("predictor lacks an exact VocabularyIdentity")
    vocabulary.__post_init__()
    if type(estimator_spec) is not EstimatorSpec:
        raise ValueError("predictor lacks an exact EstimatorSpec")
    estimator_spec.__post_init__()
    if type(estimator_identity) is not EstimatorIdentity:
        raise ValueError("predictor lacks an exact EstimatorIdentity")
    estimator_identity.__post_init__()
    if (
        stream.estimator_semantic_sha256
        != estimator_identity.semantic_sha256
        or stream.estimator_artifact_bytes_sha256
        != estimator_identity.artifact_bytes_sha256
        or stream.estimator_identity_sha256
        != estimator_identity.identity_sha256
        or estimator_spec.estimator_sha256
        != estimator_identity.semantic_sha256
        or stream.stream_sha256 != binding.estimator_stream_sha256
        or stream.stream_seed != binding.stream_seed
    ):
        raise ValueError("estimator stream does not match the predictor")
    if (
        estimator_spec.kind == "weighted_smc"
        and estimator_spec.particle_count not in WT103_PARTICLE_COUNTS
    ):
        raise ValueError("weighted WT103 scoring requires the frozen particle ladder")
    if estimator_spec.kind == "deterministic_exact":
        if binding.logical_stream_id is not None:
            raise ValueError("exact scoring cannot acquire a logical MC stream ID")
    elif binding.logical_stream_id is None:
        raise ValueError("weighted scoring requires a logical MC stream ID")

    if type(batches) is not WT103EvaluationBatches:
        raise ValueError(
            "batches must be an exact manifest-bound WT103EvaluationBatches"
        )
    batches.__post_init__()
    if batches.manifest.split != binding.split:
        raise ValueError("batch split differs from estimator stream binding")
    for batch in batches.batches:
        valid_target_mask = batch.targets != WT103_IGNORE_TARGET_ID
        if not torch.equal(valid_target_mask, batch.attention_mask):
            raise ValueError(
                "WT103 valid-target and attention masks must agree exactly"
            )
    return batches, vocabulary, estimator_spec, binding


def _validate_wt103_prediction(
    prediction: object,
    *,
    prefix: CausalPrefix,
    vocabulary: VocabularyIdentity,
) -> PriorPrediction:
    if type(prediction) is not PriorPrediction:
        raise ValueError("predictor must return an exact PriorPrediction")
    prediction.__post_init__()
    if (
        prediction.vocabulary != vocabulary
        or prediction.cache.key.prefix_sha256 != prefix.prefix_sha256
        or prediction.cache.key.prefix_tokens
        != tuple(int(value) for value in prefix.token_ids.tolist())
    ):
        raise ValueError("prior prediction does not match its target-free prefix")
    return prediction


def _wt103_prediction_pass(
    *,
    predictor: PriorPredictor,
    batches: tuple[CausalBatch, ...],
    stream: EstimatorStream,
    vocabulary: VocabularyIdentity,
    mode: Literal["cold", "warm", "reverse"],
) -> tuple[
    dict[tuple[int, int], tuple[str, str, str]],
    tuple[float, ...],
    dict[tuple[int, int], EstimatorRecord],
]:
    rows = tuple(
        (
            window_id,
            batch.inputs[row_index],
            batch.targets[row_index],
            batch.attention_mask[row_index],
        )
        for batch in batches
        for row_index, window_id in enumerate(batch.window_ids)
    )
    traversed_rows = rows if mode != "reverse" else tuple(reversed(rows))
    fingerprints: dict[tuple[int, int], tuple[str, str, str]] = {}
    estimator_records: dict[tuple[int, int], EstimatorRecord] = {}
    negative_log_terms: list[float] = []
    for window_id, inputs, targets, attention_mask in traversed_rows:
        real_count = int(torch.sum(attention_mask).item())
        positions = tuple(range(real_count))
        if mode == "reverse":
            positions = tuple(reversed(positions))
        cache: PrefixCache | None = None
        for position in positions:
            prefix = CausalPrefix.create(
                receiver_t=position + 2,
                vocabulary=vocabulary,
                token_ids=inputs[: position + 1].clone().contiguous(),
            )
            prediction = _validate_wt103_prediction(
                predictor.next_token_log_probs(
                    prefix,
                    stream,
                    cache if mode == "warm" else None,
                ),
                prefix=prefix,
                vocabulary=vocabulary,
            )
            if mode == "warm":
                cache = prediction.cache
            fingerprints[(window_id, position)] = (
                prediction.log_probs.raw_bytes_sha256,
                prediction.cache.cache_sha256,
                prediction.estimator_record.record_sha256,
            )
            estimator_records[(window_id, position)] = (
                prediction.estimator_record
            )

            # The target is intentionally read only after the target-blind call.
            target_id = int(targets[position].item())
            if target_id == WT103_IGNORE_TARGET_ID:
                continue
            if not 0 <= target_id < vocabulary.size:
                raise ValueError("target token ID falls outside the vocabulary")
            selected = float(prediction.log_probs.value()[target_id].item())
            if not math.isfinite(selected):
                raise ValueError("selected target log probability must be finite")
            negative_log_terms.append(-selected)
    return (
        fingerprints,
        tuple(negative_log_terms),
        estimator_records,
    )


def score_prior_nll(
    predictor: PriorPredictor,
    batches: Iterable[CausalBatch],
    stream: EstimatorStream,
) -> WT103NllTotals:
    """Score one complete WT103 split without exposing a target to the predictor.

    The public boundary deliberately accepts only the unchanged
    :class:`PriorPredictor`, causal batches, and one estimator stream.  It
    performs cold-cache, warm-cache, and reverse-order passes and requires
    identical raw prediction and cache records before reporting the
    corpus-summed numerator.
    """

    evaluation_batches, vocabulary, estimator_spec, binding = (
        _validate_wt103_scoring_inputs(
            predictor=predictor,
            batches=batches,
            stream=stream,
        )
    )
    owned_batches = evaluation_batches.batches
    cold, cold_terms, cold_records = _wt103_prediction_pass(
        predictor=predictor,
        batches=owned_batches,
        stream=stream,
        vocabulary=vocabulary,
        mode="cold",
    )
    warm, warm_terms, warm_records = _wt103_prediction_pass(
        predictor=predictor,
        batches=owned_batches,
        stream=stream,
        vocabulary=vocabulary,
        mode="warm",
    )
    reverse, reverse_terms, reverse_records = _wt103_prediction_pass(
        predictor=predictor,
        batches=owned_batches,
        stream=stream,
        vocabulary=vocabulary,
        mode="reverse",
    )
    if cold != warm or cold != reverse:
        raise ValueError("cold, warm, and reverse prior/cache records differ")
    if (
        tuple(
            record.record_sha256 for _, record in sorted(cold_records.items())
        )
        != tuple(
            record.record_sha256 for _, record in sorted(warm_records.items())
        )
        or tuple(
            record.record_sha256 for _, record in sorted(cold_records.items())
        )
        != tuple(
            record.record_sha256
            for _, record in sorted(reverse_records.items())
        )
    ):
        raise ValueError("cold, warm, and reverse estimator records differ")
    if (
        math.fsum(cold_terms) != math.fsum(warm_terms)
        or math.fsum(cold_terms) != math.fsum(reverse_terms)
    ):
        raise ValueError("cache/order audit changed the corpus NLL numerator")
    counted_targets = sum(batch.counted_targets for batch in owned_batches)
    if (
        len(warm_terms) != counted_targets
        or counted_targets <= 0
    ):
        raise ValueError("counted target denominator differs from causal batches")
    summed_nll = float(math.fsum(warm_terms))
    if not math.isfinite(summed_nll) or summed_nll < 0.0:
        raise ValueError("corpus-summed NLL must be finite and nonnegative")
    nll_per_token = summed_nll / counted_targets
    try:
        perplexity = math.exp(nll_per_token)
    except OverflowError as exc:
        raise ValueError("prior perplexity overflowed binary64") from exc
    if not math.isfinite(perplexity):
        raise ValueError("prior perplexity must be finite")

    scorer_kind: Literal["exact_autoregressive", "weighted_smc"]
    if estimator_spec.kind == "deterministic_exact":
        scorer_kind = "exact_autoregressive"
        estimator_stream_id = None
        particle_count = None
    else:
        scorer_kind = "weighted_smc"
        estimator_stream_id = binding.logical_stream_id
        particle_count = estimator_spec.particle_count
    audit_payload = {
        "schema_version": "wt103-target-blind-cache-audit-v1",
        "window_ids": tuple(
            window_id
            for batch in owned_batches
            for window_id in batch.window_ids
        ),
        "window_manifest_sha256": (
            evaluation_batches.manifest.manifest_sha256
        ),
        "schedule_sha256": (
            evaluation_batches.schedule.schedule_sha256
        ),
        "window_source_sha256": (
            evaluation_batches.window_source_sha256
        ),
        "batches_sha256": evaluation_batches.batches_sha256,
        "estimator_stream_binding_sha256": binding.binding_sha256,
        "estimator_stream_sha256": stream.stream_sha256,
        "common_stream_registry_sha256": (
            binding.common_stream_registry_sha256
        ),
        "prefix_records": tuple(
            (
                window_id,
                position,
                *cold[(window_id, position)],
            )
            for window_id, position in sorted(cold)
        ),
        "counted_targets": counted_targets,
    }
    payload = {
        "schema_version": "wt103-nll-totals-v1",
        "scorer_kind": scorer_kind,
        "summed_nll": summed_nll,
        "counted_targets": counted_targets,
        "nll_per_token": nll_per_token,
        "perplexity": perplexity,
        "estimator_stream_id": estimator_stream_id,
        "particle_count": particle_count,
        "cache_audit_sha256": owned_sha256(
            "vfe4.wt103.target-blind-cache-audit.v1",
            audit_payload,
        ),
    }
    totals = WT103NllTotals(
        **payload,
        totals_sha256=owned_sha256(
            "vfe4.wt103.nll-totals.v1",
            payload,
        ),
    )
    trace = WT103ScoreTrace.create(
        evaluation_batches=evaluation_batches,
        binding=binding,
        stream=stream,
        totals=totals,
        estimator_records=tuple(
            record for _, record in sorted(warm_records.items())
        ),
        negative_log_terms=warm_terms,
    )
    _ISSUED_SCORE_TRACES[id(totals)] = _IssuedScoreTrace(
        totals=totals,
        trace=trace,
    )
    return totals


__all__ = [
    "WT103EstimatorStreamBinding",
    "WT103EvaluationBatches",
    "WT103ScoreTrace",
    "bind_wt103_prior_predictor",
    "score_prior_nll",
    "score_bounded_prior_nll_replicate_v3",
    "score_prior_nll_replicate",
    "wt103_common_stream_registry_sha256",
    "wt103_estimator_stream_seed",
    "wt103_score_trace",
]
