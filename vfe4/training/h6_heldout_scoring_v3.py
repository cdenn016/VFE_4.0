"""Canonical, target-blind H6-Prediction v3 held-out scoring."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from vfe4.artifacts.h6_prediction_v3 import (
    H6CheckpointCandidateV3,
    H6ExactA0CorpusTotalV3,
    H6RawEndpointInventoryV4,
    H6WeightedA5CorpusTotalV3,
    H6_WEIGHTED_COMMON_STREAM_ROOT_SEED,
    h6_weighted_common_stream_sha256_v3,
)
from vfe4.data.windows import CausalPrefix, CausalWindows
from vfe4.evaluation.prior_nll import score_bounded_prior_nll_replicate_v3
from vfe4.evaluation.smc_uncertainty import (
    EndpointSmcAggregate,
    EndpointSmcObservation,
    aggregate_endpoint_smc,
)
from vfe4.predictive import EstimatorStream
from vfe4.training.arms import (
    ArmModel,
    LatentLanguageArmModel,
    _model_family_sha256,
    _predictive_boundary,
)
from vfe4.training.h6_experiment_v3 import H6_CONFIRMATORY_SEEDS_V3
from vfe4.training.h6_matching_v3 import H6_MATCHING_V3_ENDPOINT_CONFIG_IDS
from vfe4.training.h6_transformer import H6CausalTransformer
from vfe4.training.h6_validation_v3 import H6EvaluationArmV3
from vfe4.types import (
    H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS,
    A0DirectExactPrefixCertificateV1,
    ArmConfig,
    ArmId,
    BoundedPrefixCertificate,
    BoundedPrefixCertificateSet,
    EstimatorSpec,
    EvidenceStatus,
    GateStatus,
    H6BoundedPrefixGateResult,
    H6PredictionV3ReadinessToken,
    NllTotals,
)


_WEIGHTED_ENDPOINT_CONFIG_IDS = frozenset(
    (
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5],
        H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9],
    )
)
_A0_ENDPOINT_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[0]
_COMPLETE_A5_ENDPOINT_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[5]
_EMISSION_A5_ENDPOINT_CONFIG_ID = H6_MATCHING_V3_ENDPOINT_CONFIG_IDS[9]
_HELDOUT_ENDPOINT_CONFIG_IDS = (
    _A0_ENDPOINT_CONFIG_ID,
    _COMPLETE_A5_ENDPOINT_CONFIG_ID,
    _EMISSION_A5_ENDPOINT_CONFIG_ID,
)
_PARTICLE_COUNTS = (128, 256, 512, 1024)
_REPLICATES = tuple(range(64))
_STREAM_SEED_DOMAIN = b"VFE4-H6-WEIGHTED-ESTIMATOR-STREAM-SEED-V3\x00"


def _require_cpu_evaluation_model(model: ArmModel) -> None:
    if model.training:
        raise ValueError("held-out evaluation model must be in eval mode")
    for parameter in model.parameters():
        if (
            parameter.device.type != "cpu"
            or parameter.dtype is not torch.float64
            or parameter.requires_grad
        ):
            raise ValueError(
                "held-out evaluation parameters must be frozen CPU float64"
            )
    for buffer in model.buffers():
        if buffer.device.type != "cpu" or (
            buffer.is_floating_point() and buffer.dtype is not torch.float64
        ):
            raise ValueError("held-out evaluation buffers must be CPU float64")


def _require_log_probs(value: object, *, vocabulary_size: int) -> Tensor:
    if (
        type(value) is not Tensor
        or value.device.type != "cpu"
        or value.dtype is not torch.float64
        or tuple(value.shape) != (vocabulary_size,)
        or not bool(torch.isfinite(value).all().item())
        or not bool(
            torch.isclose(
                torch.logsumexp(value, dim=0),
                torch.tensor(0.0, dtype=torch.float64),
                atol=1.0e-10,
                rtol=0.0,
            ).item()
        )
    ):
        raise ValueError(
            "exact A0 prior must return normalized finite CPU float64 log probabilities"
        )
    return value


def _validate_direct_a0_certificate(
    certificate: object,
    *,
    readiness: H6PredictionV3ReadinessToken | None = None,
    config: ArmConfig | None = None,
    model: H6CausalTransformer | None = None,
) -> A0DirectExactPrefixCertificateV1:
    """Validate direct-A0 authority before any held-out data access."""

    if type(certificate) is not A0DirectExactPrefixCertificateV1:
        raise ValueError(
            "exact A0 scoring requires an exact direct-A0 Prefix certificate"
        )
    certificate.__post_init__()
    if (
        certificate.status is not EvidenceStatus.PASS
        or certificate.obligations != ()
        or tuple(certificate.checks)
        != H6_A0_DIRECT_EXACT_PREFIX_REQUIRED_CHECKS
        or not all(certificate.checks.values())
    ):
        raise ValueError(
            "exact A0 scoring requires a closed PASS direct-A0 certificate"
        )
    if readiness is not None:
        if type(readiness) is not H6PredictionV3ReadinessToken:
            raise ValueError(
                "exact A0 scoring requires exact v3 readiness"
            )
        readiness.__post_init__()
        if (
            readiness.status != "PASS"
            or certificate.certificate_sha256
            != readiness.a0_direct_exact_prefix_certificate_sha256
            or certificate.git_head != readiness.git_head
            or certificate.dirty_digest != readiness.dirty_digest
        ):
            raise ValueError(
                "direct-A0 certificate differs from v3 readiness"
            )
    if config is not None or model is not None:
        if type(config) is not ArmConfig:
            raise ValueError("exact A0 scoring requires an exact ArmConfig")
        config.__post_init__()
        if (
            config.arm is not ArmId.A0
            or config.config_id != _A0_ENDPOINT_CONFIG_ID
            or type(model) is not H6CausalTransformer
            or model.vocabulary != config.vocabulary
            or certificate.endpoint_config != config
            or certificate.predictor_config_sha256 != config.config_sha256
            or certificate.model_family_sha256
            != _model_family_sha256(config)
        ):
            raise ValueError(
                "direct-A0 certificate differs from the held-out endpoint"
            )
    return certificate


def score_h6_exact_a0_total_v3(
    *,
    config: ArmConfig,
    model: H6CausalTransformer,
    windows: CausalWindows,
    certificate: A0DirectExactPrefixCertificateV1,
) -> NllTotals:
    """Score one A0 checkpoint over authorized test windows exactly once."""

    _validate_direct_a0_certificate(
        certificate,
        config=config,
        model=model,
    )
    _require_cpu_evaluation_model(model)
    if type(windows) is not CausalWindows or windows.split != "test":
        raise ValueError("held-out scoring requires exact test CausalWindows")
    windows.__post_init__()

    losses: list[float] = []
    with torch.no_grad():
        for window_index, real_target_count in enumerate(
            windows.real_target_counts
        ):
            observed_history: list[int] = []
            for position in range(real_target_count):
                prefix = CausalPrefix.create(
                    receiver_t=len(observed_history) + 1,
                    vocabulary=model.vocabulary,
                    token_ids=torch.tensor(
                        observed_history,
                        dtype=torch.int64,
                        device="cpu",
                    ),
                )
                log_probs = _require_log_probs(
                    model.prefix_log_probs(prefix),
                    vocabulary_size=model.vocabulary.size,
                )
                # The target is intentionally read only after the target-blind call.
                target = windows.targets[window_index][position]
                loss = -float(log_probs[target].item())
                if not math.isfinite(loss) or loss < 0.0:
                    raise ValueError("exact A0 target NLL must be finite/nonnegative")
                losses.append(loss)
                observed_history.append(target)
    if len(losses) != windows.counted_target_total:
        raise ValueError("exact A0 target accounting drift")
    return NllTotals(
        negative_log_likelihood_sum=float(math.fsum(losses)),
        counted_targets=len(losses),
    )


def h6_weighted_estimator_stream_seed_v3(*, replicate_id: int) -> int:
    """Derive one public unsigned-64 counter seed from the frozen registry."""

    if type(replicate_id) is not int or replicate_id not in _REPLICATES:
        raise ValueError("replicate_id is outside the frozen 64 streams")
    common_stream_sha256 = h6_weighted_common_stream_sha256_v3(
        replicate_id=replicate_id
    )
    digest = hashlib.sha256(
        _STREAM_SEED_DOMAIN
        + H6_WEIGHTED_COMMON_STREAM_ROOT_SEED.to_bytes(
            8,
            "little",
            signed=False,
        )
        + replicate_id.to_bytes(8, "little", signed=False)
        + bytes.fromhex(common_stream_sha256)
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def score_h6_weighted_a5_total_v3(
    *,
    config: ArmConfig,
    model: LatentLanguageArmModel,
    windows: CausalWindows,
    particle_count: Literal[128, 256, 512, 1024],
    replicate_id: int,
    certificate: BoundedPrefixCertificate,
) -> NllTotals:
    """Score one frozen common-stream/particle A5 corpus total."""

    if type(config) is not ArmConfig:
        raise ValueError("weighted A5 scoring requires an exact ArmConfig")
    config.__post_init__()
    if (
        config.arm is not ArmId.A5
        or config.config_id not in _WEIGHTED_ENDPOINT_CONFIG_IDS
        or type(model) is not LatentLanguageArmModel
    ):
        raise ValueError(
            "weighted held-out scoring is limited to the two frozen A5 endpoints"
        )
    if model.arm is not ArmId.A5 or model.vocabulary != config.vocabulary:
        raise ValueError("weighted A5 model does not match its endpoint config")
    _require_cpu_evaluation_model(model)
    if type(windows) is not CausalWindows or windows.split != "test":
        raise ValueError("held-out scoring requires exact test CausalWindows")
    windows.__post_init__()
    if type(particle_count) is not int or particle_count not in _PARTICLE_COUNTS:
        raise ValueError("particle_count is outside the frozen endpoint ladder")
    stream_seed = h6_weighted_estimator_stream_seed_v3(
        replicate_id=replicate_id
    )
    if type(certificate) is not BoundedPrefixCertificate:
        raise ValueError(
            "weighted A5 scoring requires an exact BoundedPrefixCertificate"
        )
    certificate.__post_init__()

    estimator_spec = EstimatorSpec.create(
        kind="weighted_smc",
        particle_count=particle_count,
        resampling="systematic_ess_half",
    )
    _, predictor = _predictive_boundary(
        config=config,
        model=model,
        model_family_sha256=_model_family_sha256(config),
        estimator_spec=estimator_spec,
    )
    stream = EstimatorStream.create(
        stream_seed=stream_seed,
        estimator_identity=predictor.estimator_identity,
    )
    with torch.no_grad():
        return score_bounded_prior_nll_replicate_v3(
            predictor,
            windows,
            stream,
            particle_count,
            certificate,
        )


@dataclass(frozen=True, slots=True)
class H6HeldoutCheckpointArmV3:
    """One selected checkpoint bound to its exact byte-validated evaluation arm."""

    candidate: H6CheckpointCandidateV3
    evaluation: H6EvaluationArmV3

    def __post_init__(self) -> None:
        if type(self.candidate) is not H6CheckpointCandidateV3:
            raise ValueError("held-out binding requires an exact checkpoint candidate")
        if type(self.evaluation) is not H6EvaluationArmV3:
            raise ValueError("held-out binding requires an exact evaluation arm")
        self.candidate.__post_init__()
        self.evaluation.__post_init__()
        if self.evaluation.evaluation_role != "heldout":
            raise ValueError("held-out binding requires the heldout evaluation role")
        for name in (
            "checkpoint_sha256",
            "checkpoint_bytes_sha256",
            "planned_attempt_sha256",
            "attempt_spec_sha256",
            "endpoint_config_id",
            "endpoint_config_sha256",
            "training_seed",
        ):
            if getattr(self.candidate, name) != getattr(self.evaluation, name):
                raise ValueError(
                    f"held-out checkpoint/evaluation {name} authority drift"
                )
        endpoint_id = self.evaluation.endpoint_config_id
        model = self.evaluation.model
        if endpoint_id not in _HELDOUT_ENDPOINT_CONFIG_IDS:
            raise ValueError(
                "held-out binding is limited to A0 and the two frozen A5 endpoints"
            )
        if (
            endpoint_id == _A0_ENDPOINT_CONFIG_ID
            and type(model) is not H6CausalTransformer
        ) or (
            endpoint_id in _WEIGHTED_ENDPOINT_CONFIG_IDS
            and type(model) is not LatentLanguageArmModel
        ):
            raise ValueError("held-out evaluation model family is wrong")


def _validate_checkpoint_arms(
    checkpoint_arms: object,
) -> tuple[H6HeldoutCheckpointArmV3, ...]:
    if (
        type(checkpoint_arms) is not tuple
        or len(checkpoint_arms) != 24
        or any(
            type(binding) is not H6HeldoutCheckpointArmV3
            for binding in checkpoint_arms
        )
    ):
        raise ValueError("held-out scorer requires exactly 24 checkpoint arms")
    expected_keys = tuple(
        (endpoint_id, seed)
        for endpoint_id in _HELDOUT_ENDPOINT_CONFIG_IDS
        for seed in H6_CONFIRMATORY_SEEDS_V3
    )
    observed_keys = []
    for binding in checkpoint_arms:
        binding.__post_init__()
        observed_keys.append(
            (
                binding.evaluation.endpoint_config_id,
                binding.evaluation.training_seed,
            )
        )
    if tuple(observed_keys) != expected_keys:
        raise ValueError(
            "held-out checkpoint arms are incomplete, duplicated, or out of order"
        )
    return checkpoint_arms


def _validated_bounded_certificate_set(
    prefix_certificate_set: object,
    *,
    readiness: H6PredictionV3ReadinessToken,
) -> tuple[BoundedPrefixCertificate, ...]:
    if type(readiness) is not H6PredictionV3ReadinessToken:
        raise ValueError(
            "held-out scorer requires an exact v3 readiness authority"
        )
    readiness.__post_init__()
    if type(prefix_certificate_set) is not BoundedPrefixCertificateSet:
        raise ValueError(
            "held-out scorer requires an exact BoundedPrefixCertificateSet"
        )
    prefix_certificate_set.__post_init__()
    gate = H6BoundedPrefixGateResult.from_certificate_set(
        prefix_certificate_set
    )
    if (
        gate.status is not GateStatus.PASS
        or gate.obligations != ()
        or gate.prefix_certificate_set_sha256
        != readiness.prefix_certificate_set_sha256
        or prefix_certificate_set.git_head != readiness.git_head
        or prefix_certificate_set.dirty_digest != readiness.dirty_digest
    ):
        raise ValueError(
            "held-out bounded Prefix certificate set is not exact current PASS"
        )
    return prefix_certificate_set.certificates


def _required_weighted_certificates(
    *,
    bindings: tuple[H6HeldoutCheckpointArmV3, ...],
    certificates: tuple[BoundedPrefixCertificate, ...],
) -> dict[tuple[str, int], BoundedPrefixCertificate]:
    result: dict[tuple[str, int], BoundedPrefixCertificate] = {}
    representative_evaluations = {
        binding.evaluation.endpoint_config_id: binding.evaluation
        for binding in bindings
        if binding.evaluation.endpoint_config_id in _WEIGHTED_ENDPOINT_CONFIG_IDS
    }
    for endpoint_id in (
        _COMPLETE_A5_ENDPOINT_CONFIG_ID,
        _EMISSION_A5_ENDPOINT_CONFIG_ID,
    ):
        evaluation = representative_evaluations[endpoint_id]
        config = evaluation.config
        model = evaluation.model
        if type(model) is not LatentLanguageArmModel:
            raise ValueError("weighted evaluation model family changed")
        for particle_count in _PARTICLE_COUNTS:
            estimator_spec = EstimatorSpec.create(
                kind="weighted_smc",
                particle_count=particle_count,
                resampling="systematic_ess_half",
            )
            _, predictor = _predictive_boundary(
                config=config,
                model=model,
                model_family_sha256=_model_family_sha256(config),
                estimator_spec=estimator_spec,
            )
            matching = tuple(
                certificate
                for certificate in certificates
                if any(
                    (
                        reference.particle_count == particle_count
                        and reference.case_family == "validation"
                        and reference.report_key.arm is ArmId.A5
                        and reference.report_key.predictor_config_sha256
                        == predictor.predictor_config_sha256
                        and reference.report_key.estimator_sha256
                        == predictor.estimator_spec.estimator_sha256
                        and reference.report_key.model_family_sha256
                        == predictor.model_family_sha256
                        and reference.report_key.vocabulary_sha256
                        == predictor.vocabulary_sha256
                        and reference.report_key.data_safety_sha256
                        == predictor.data_safety_sha256
                    )
                    for reference in (
                        certificate.report_binding.report_references
                    )
                )
            )
            if len(matching) != 1:
                raise ValueError(
                    "held-out scorer lacks one unique weighted PrefixCertificate"
                )
            certificate = matching[0]
            result[(endpoint_id, particle_count)] = certificate
    return result


def _weighted_rows_for_binding(
    *,
    binding: H6HeldoutCheckpointArmV3,
    windows: CausalWindows,
    opening_proof_sha256: str,
    endpoint_role: Literal["complete_a5", "emission_a5"],
    certificates: dict[tuple[str, int], BoundedPrefixCertificate],
) -> tuple[H6WeightedA5CorpusTotalV3, ...]:
    evaluation = binding.evaluation
    model = evaluation.model
    if type(model) is not LatentLanguageArmModel:
        raise ValueError("weighted binding does not carry an exact latent A5 model")
    scored: list[tuple[int, int, NllTotals]] = []
    observations: list[EndpointSmcObservation] = []
    for replicate_id in _REPLICATES:
        common_stream_sha256 = h6_weighted_common_stream_sha256_v3(
            replicate_id=replicate_id
        )
        for particle_count in _PARTICLE_COUNTS:
            totals = score_h6_weighted_a5_total_v3(
                config=evaluation.config,
                model=model,
                windows=windows,
                particle_count=particle_count,
                replicate_id=replicate_id,
                certificate=certificates[
                    (evaluation.endpoint_config_id, particle_count)
                ],
            )
            if type(totals) is not NllTotals:
                raise ValueError("weighted scorer did not return exact NLL totals")
            totals.__post_init__()
            scored.append((replicate_id, particle_count, totals))
            observations.append(
                EndpointSmcObservation(
                    checkpoint_sha256=binding.candidate.checkpoint_sha256,
                    replicate_id=replicate_id,
                    particle_count=particle_count,
                    common_stream_sha256=common_stream_sha256,
                    negative_log_likelihood_sum=(
                        totals.negative_log_likelihood_sum
                    ),
                    counted_targets=totals.counted_targets,
                )
            )
    aggregate = aggregate_endpoint_smc(observations)
    if (
        type(aggregate) is not EndpointSmcAggregate
        or aggregate.checkpoint_sha256
        != binding.candidate.checkpoint_sha256
        or aggregate.status is not EvidenceStatus.PASS
        or aggregate.eligible is not True
        or aggregate.obligations != ()
    ):
        raise ValueError(
            "complete weighted A5 uncertainty arithmetic is not eligible PASS"
        )
    return tuple(
        H6WeightedA5CorpusTotalV3.create(
            endpoint_role=endpoint_role,
            endpoint_config_id=evaluation.endpoint_config_id,
            training_seed=evaluation.training_seed,
            checkpoint_sha256=binding.candidate.checkpoint_sha256,
            particle_count=particle_count,
            replicate_id=replicate_id,
            counted_test_targets=totals.counted_targets,
            weighted_total_nll=totals.negative_log_likelihood_sum,
            monte_carlo_half_width=aggregate.half_width,
            smc_bias_bound=aggregate.bias_bound,
            opening_proof_sha256=opening_proof_sha256,
        )
        for replicate_id, particle_count, totals in scored
    )


def score_h6_heldout_inventory_v3(
    *,
    windows: CausalWindows,
    opening_proof_sha256: str,
    checkpoint_arms: tuple[H6HeldoutCheckpointArmV3, ...],
    prefix_certificate_set: BoundedPrefixCertificateSet,
    a0_direct_exact_prefix_certificate: (
        A0DirectExactPrefixCertificateV1
    ),
    readiness: H6PredictionV3ReadinessToken,
) -> H6RawEndpointInventoryV4:
    """Score the sole exact 4,104-row H6 held-out inventory."""

    direct_certificate = _validate_direct_a0_certificate(
        a0_direct_exact_prefix_certificate,
        readiness=readiness,
    )
    if type(windows) is not CausalWindows or windows.split != "test":
        raise ValueError("held-out scoring requires exact test CausalWindows")
    windows.__post_init__()
    if (
        type(opening_proof_sha256) is not str
        or len(opening_proof_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in opening_proof_sha256
        )
    ):
        raise ValueError("opening_proof_sha256 must be a lowercase SHA-256")
    bindings = _validate_checkpoint_arms(checkpoint_arms)
    certificates = _required_weighted_certificates(
        bindings=bindings,
        certificates=_validated_bounded_certificate_set(
            prefix_certificate_set,
            readiness=readiness,
        ),
    )

    a0_bindings = bindings[:8]
    complete_bindings = bindings[8:16]
    emission_bindings = bindings[16:24]
    exact_rows = []
    for binding in a0_bindings:
        model = binding.evaluation.model
        if type(model) is not H6CausalTransformer:
            raise ValueError("exact A0 binding changed after validation")
        totals = score_h6_exact_a0_total_v3(
            config=binding.evaluation.config,
            model=model,
            windows=windows,
            certificate=direct_certificate,
        )
        if type(totals) is not NllTotals:
            raise ValueError("exact A0 scorer did not return exact NLL totals")
        totals.__post_init__()
        exact_rows.append(
            H6ExactA0CorpusTotalV3.create(
                endpoint_config_id=binding.evaluation.endpoint_config_id,
                training_seed=binding.evaluation.training_seed,
                checkpoint_sha256=binding.candidate.checkpoint_sha256,
                counted_test_targets=totals.counted_targets,
                exact_total_nll=totals.negative_log_likelihood_sum,
                opening_proof_sha256=opening_proof_sha256,
            )
        )
    complete_rows = tuple(
        row
        for binding in complete_bindings
        for row in _weighted_rows_for_binding(
            binding=binding,
            windows=windows,
            opening_proof_sha256=opening_proof_sha256,
            endpoint_role="complete_a5",
            certificates=certificates,
        )
    )
    emission_rows = tuple(
        row
        for binding in emission_bindings
        for row in _weighted_rows_for_binding(
            binding=binding,
            windows=windows,
            opening_proof_sha256=opening_proof_sha256,
            endpoint_role="emission_a5",
            certificates=certificates,
        )
    )
    return H6RawEndpointInventoryV4.create(
        exact_a0_rows=tuple(exact_rows),
        complete_a5_rows=complete_rows,
        emission_a5_rows=emission_rows,
    )


__all__ = [
    "H6HeldoutCheckpointArmV3",
    "h6_weighted_estimator_stream_seed_v3",
    "score_h6_exact_a0_total_v3",
    "score_h6_heldout_inventory_v3",
    "score_h6_weighted_a5_total_v3",
]
