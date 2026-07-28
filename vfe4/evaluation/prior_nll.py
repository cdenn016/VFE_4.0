"""Target-blind corpus-summed prior-NLL scoring for H6."""

from __future__ import annotations

import math
from typing import Literal

import torch

from vfe4.data.windows import CausalPrefix, CausalWindows, SEQUENCE_LENGTH
from vfe4.predictive import (
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
    VocabularyIdentity,
)


_PARTICLE_COUNTS = (128, 256, 512, 1024)


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


__all__ = [
    "score_bounded_prior_nll_replicate_v3",
    "score_prior_nll_replicate",
]
