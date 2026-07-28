"""Executable mixed recognition ELBO for H6 Prediction v3.

This evaluator is intentionally independent of the historical marker-only
source-law and one-Gaussian objective adapters.  It consumes the receiver
trajectory directly, sums every finite source row exactly, and leaves one live
scalar for reverse-mode autograd over the phase-owned parameter block.
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

from vfe4.data.windows import CausalPrefix
from vfe4.recognition.h6_prediction_v3 import (
    AbsentSourceBank,
    CategoricalSourceBank,
    CategoricalSourceRow,
    GaussianReceiverComponent,
    LanguageRecognitionTrajectory,
    SourceBankName,
)

if TYPE_CHECKING:
    from vfe4.training.arms import LatentLanguageArmModel


H6MixtureModeV3 = Literal["exact", "moment_projection"]
H6ActiveParameterBlockV3 = Literal["recognition", "model"]
H6ObjectivePartitionV3 = Literal[
    "initial",
    "state_source",
    "model_source",
    "state_transition",
    "model_transition",
    "emission",
    "gaussian_entropy",
]

_LOWER_HEX = frozenset("0123456789abcdef")


def _tensor_fingerprint(value: Tensor) -> dict[str, object]:
    cpu = value.detach().to(device="cpu").contiguous().reshape(-1)
    raw = cpu.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _identity(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii")
        + b"\x00"
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_float64(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype is not torch.float64
        or (ndim is not None and value.ndim != ndim)
        or (shape is not None and tuple(value.shape) != shape)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"{name} must be a finite float64 tensor of the required shape"
        )
    return value


def _require_scalar(value: object, name: str) -> Tensor:
    return _require_float64(value, name=name, shape=())


@dataclass(frozen=True, slots=True, eq=False)
class H6ObjectiveTermV3:
    """One graph-live canonical objective slot."""

    partition: H6ObjectivePartitionV3
    receiver_t: int
    value: Tensor
    term_identity_sha256: str

    def __post_init__(self) -> None:
        if self.partition not in (
            "initial",
            "state_source",
            "model_source",
            "state_transition",
            "model_transition",
            "emission",
            "gaussian_entropy",
        ):
            raise ValueError("unsupported H6 v3 objective partition")
        if type(self.receiver_t) is not int or self.receiver_t < 0:
            raise ValueError("receiver_t must be a nonnegative integer")
        _require_scalar(self.value, "objective term")
        _require_sha256(self.term_identity_sha256, "term_identity_sha256")


@dataclass(frozen=True, slots=True, eq=False)
class H6SourceRowEvaluationV3:
    """One exact categorical source-row reduction on sampled history."""

    bank: SourceBankName
    receiver_t: int
    support: tuple[int, ...]
    posterior_log_probabilities: Tensor
    generative_log_prior: Tensor
    transition_log_probabilities: Tensor
    sampled_earlier_latents: Tensor
    categorical_entropy: Tensor
    source_log_prior_contribution: Tensor
    transition_contribution: Tensor
    entropy_contribution: Tensor
    combined_reduction: Tensor
    row_evaluation_identity_sha256: str

    def __post_init__(self) -> None:
        if self.bank not in ("state", "model"):
            raise ValueError("source-row bank must be state or model")
        if (
            type(self.receiver_t) is not int
            or self.receiver_t <= 0
            or type(self.support) is not tuple
            or not self.support
            or any(
                type(source_j) is not int or source_j < 0 or source_j >= self.receiver_t
                for source_j in self.support
            )
        ):
            raise ValueError("source-row support must be strictly causal")
        width = len(self.support)
        posterior = _require_float64(
            self.posterior_log_probabilities,
            name="posterior_log_probabilities",
            shape=(width,),
        )
        prior = _require_float64(
            self.generative_log_prior,
            name="generative_log_prior",
            shape=(width,),
        )
        transition = _require_float64(
            self.transition_log_probabilities,
            name="transition_log_probabilities",
            shape=(width,),
        )
        history = _require_float64(
            self.sampled_earlier_latents,
            name="sampled_earlier_latents",
            ndim=2,
        )
        if history.shape[0] != self.receiver_t:
            raise ValueError("sampled source history must cover vertices 0..t-1")
        if (
            len({posterior.device, prior.device, transition.device, history.device})
            != 1
        ):
            raise ValueError("source-row tensors must share one device")
        for name in (
            "categorical_entropy",
            "source_log_prior_contribution",
            "transition_contribution",
            "entropy_contribution",
            "combined_reduction",
        ):
            _require_scalar(getattr(self, name), name)
        _require_sha256(
            self.row_evaluation_identity_sha256,
            "row_evaluation_identity_sha256",
        )


@dataclass(frozen=True, slots=True, eq=False)
class H6TerminalComponentEvaluationV3:
    """One explicitly evaluated terminal ``(j,k)`` source component."""

    state_source_j: int | None
    model_source_j: int | None
    posterior_log_weight: Tensor
    sample: Tensor
    state_log_prior: Tensor
    state_transition_log_prob: Tensor
    model_log_prior: Tensor
    model_transition_log_prob: Tensor
    emission_log_prob: Tensor
    categorical_log_ratio: Tensor
    bracket: Tensor
    weighted_contribution: Tensor
    component_evaluation_identity_sha256: str

    def __post_init__(self) -> None:
        for source_j in (self.state_source_j, self.model_source_j):
            if source_j is not None and (type(source_j) is not int or source_j < 0):
                raise ValueError("terminal source labels must be nonnegative")
        _require_float64(self.sample, name="terminal sample", ndim=1)
        for name in (
            "posterior_log_weight",
            "state_log_prior",
            "state_transition_log_prob",
            "model_log_prior",
            "model_transition_log_prob",
            "emission_log_prob",
            "categorical_log_ratio",
            "bracket",
            "weighted_contribution",
        ):
            _require_scalar(getattr(self, name), name)
        _require_sha256(
            self.component_evaluation_identity_sha256,
            "component_evaluation_identity_sha256",
        )


@dataclass(frozen=True, slots=True, eq=False)
class ExactSourceMixtureEvaluationV3:
    """Evaluated exact source law, never an endpoint-identity marker."""

    trajectory_identity_sha256: str
    source_independent_samples: tuple[Tensor, ...]
    source_rows: tuple[H6SourceRowEvaluationV3, ...]
    terminal_components: tuple[H6TerminalComponentEvaluationV3, ...]
    shared_precision_cholesky: Tensor
    component_gaussian_entropy: Tensor
    law_identity_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(
            self.trajectory_identity_sha256,
            "trajectory_identity_sha256",
        )
        if (
            type(self.source_independent_samples) is not tuple
            or not self.source_independent_samples
            or any(
                not isinstance(value, Tensor) or value.ndim != 1
                for value in self.source_independent_samples
            )
            or type(self.source_rows) is not tuple
            or type(self.terminal_components) is not tuple
            or not self.terminal_components
        ):
            raise ValueError("exact source evaluation inventory is incomplete")
        _require_float64(
            self.shared_precision_cholesky,
            name="shared_precision_cholesky",
            ndim=2,
        )
        _require_scalar(
            self.component_gaussian_entropy,
            "component_gaussian_entropy",
        )
        _require_sha256(self.law_identity_sha256, "law_identity_sha256")


@dataclass(frozen=True, slots=True, eq=False)
class MomentProjectionEvaluationV3:
    """Live first-two-moment projection with its labeled dispersion bound."""

    component_weights: Tensor
    component_means: Tensor
    shared_component_covariance: Tensor
    projected_mean: Tensor
    projected_covariance: Tensor
    projected_cholesky: Tensor
    sample: Tensor
    analytic_entropy: Tensor
    component_kl_upper_bound: Tensor
    projection_identity_sha256: str
    bound_label: Literal["weighted_component_kl_upper_bound"] = (
        "weighted_component_kl_upper_bound"
    )

    def __post_init__(self) -> None:
        weights = _require_float64(
            self.component_weights,
            name="component_weights",
            ndim=1,
        )
        means = _require_float64(
            self.component_means,
            name="component_means",
            ndim=2,
        )
        if means.shape[0] != weights.numel() or means.shape[0] <= 0:
            raise ValueError("projection weights and component means must align")
        dimension = means.shape[1]
        for name in (
            "shared_component_covariance",
            "projected_covariance",
            "projected_cholesky",
        ):
            _require_float64(
                getattr(self, name),
                name=name,
                shape=(dimension, dimension),
            )
        for name in ("projected_mean", "sample"):
            _require_float64(
                getattr(self, name),
                name=name,
                shape=(dimension,),
            )
        for name in ("analytic_entropy", "component_kl_upper_bound"):
            _require_scalar(getattr(self, name), name)
        if not bool(self.component_kl_upper_bound >= 0.0):
            raise ValueError("component-KL upper bound must be nonnegative")
        if self.bound_label != "weighted_component_kl_upper_bound":
            raise ValueError("projection bound must not be relabeled as mixture KL")
        _require_sha256(
            self.projection_identity_sha256,
            "projection_identity_sha256",
        )


H6EvaluatedRecognitionLawV3 = (
    ExactSourceMixtureEvaluationV3 | MomentProjectionEvaluationV3
)


@dataclass(frozen=True, slots=True, eq=False)
class H6PredictionObjectiveEstimateV3:
    """One graph-live positive ELBO estimate and its minimizing loss."""

    mixture_mode: H6MixtureModeV3
    active_parameter_block: H6ActiveParameterBlockV3
    source_law: H6EvaluatedRecognitionLawV3
    evaluated_source_rows: tuple[H6SourceRowEvaluationV3, ...]
    ordered_terms: tuple[H6ObjectiveTermV3, ...]
    terminal_joint_contribution: Tensor
    canonical_ordered_total: Tensor
    independently_accumulated_total: Tensor
    elbo: Tensor
    loss: Tensor
    estimate_identity_sha256: str

    def __post_init__(self) -> None:
        if self.mixture_mode not in ("exact", "moment_projection"):
            raise ValueError("unsupported H6 v3 mixture mode")
        if self.active_parameter_block not in ("recognition", "model"):
            raise ValueError("unsupported H6 v3 active parameter block")
        if (
            type(self.ordered_terms) is not tuple
            or not self.ordered_terms
            or any(type(term) is not H6ObjectiveTermV3 for term in self.ordered_terms)
        ):
            raise ValueError("objective ordered term inventory is incomplete")
        for name in (
            "terminal_joint_contribution",
            "canonical_ordered_total",
            "independently_accumulated_total",
            "elbo",
            "loss",
        ):
            _require_scalar(getattr(self, name), name)
        if (
            self.canonical_ordered_total is not self.independently_accumulated_total
            or self.elbo is not self.canonical_ordered_total
        ):
            raise ValueError(
                "canonical, independent, and exposed ELBO totals must be one live scalar"
            )
        if not bool(torch.equal(self.loss, -self.elbo)):
            raise ValueError("H6 v3 optimizers must minimize loss = -ELBO")
        _require_sha256(
            self.estimate_identity_sha256,
            "estimate_identity_sha256",
        )


def _covariance_from_precision_cholesky(precision_cholesky: Tensor) -> Tensor:
    dimension = precision_cholesky.shape[0]
    identity = torch.eye(
        dimension,
        dtype=precision_cholesky.dtype,
        device=precision_cholesky.device,
    )
    inverse = torch.linalg.solve_triangular(
        precision_cholesky,
        identity,
        upper=False,
    )
    return inverse.transpose(-1, -2) @ inverse


def _sample_precision_gaussian(
    mean: Tensor,
    precision_cholesky: Tensor,
    base_noise: Tensor,
) -> Tensor:
    displacement = torch.linalg.solve_triangular(
        precision_cholesky.transpose(-1, -2),
        base_noise.unsqueeze(-1),
        upper=True,
    ).squeeze(-1)
    return mean + displacement


def _precision_gaussian_entropy(precision_cholesky: Tensor) -> Tensor:
    dimension = precision_cholesky.shape[0]
    return (
        0.5 * dimension * (1.0 + math.log(2.0 * math.pi))
        - torch.log(torch.diagonal(precision_cholesky)).sum()
    )


def project_terminal_mixture_v3(
    *,
    terminal_components: tuple[GaussianReceiverComponent, ...],
    base_noise: Tensor,
) -> MomentProjectionEvaluationV3:
    """Moment-match a terminal Gaussian mixture and sample its own Cholesky."""

    if (
        type(terminal_components) is not tuple
        or not terminal_components
        or any(
            type(component) is not GaussianReceiverComponent
            for component in terminal_components
        )
    ):
        raise ValueError("terminal_components must be a nonempty exact tuple")
    first = terminal_components[0]
    first.__post_init__()
    precision = first.precision_cholesky
    dimension = first.mean.numel()
    noise = _require_float64(
        base_noise,
        name="projection base_noise",
        shape=(dimension,),
    )
    if noise.device != precision.device or noise.requires_grad:
        raise ValueError(
            "projection base_noise must be graph-free on the component device"
        )
    for component in terminal_components:
        component.__post_init__()
        if (
            component.receiver_t != first.receiver_t
            or component.precision_identity_sha256 != first.precision_identity_sha256
            or component.mean.device != precision.device
        ):
            raise ValueError(
                "terminal projection requires one receiver and shared precision"
            )
    log_weights = torch.stack(
        tuple(component.log_probability for component in terminal_components)
    )
    if not bool(
        torch.allclose(
            torch.logsumexp(log_weights, dim=0),
            log_weights.new_zeros(()),
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError("terminal component weights must be normalized")
    weights = log_weights.exp()
    means = torch.stack(tuple(component.mean for component in terminal_components))
    projected_mean = torch.sum(weights.unsqueeze(1) * means, dim=0)
    shared_covariance = _covariance_from_precision_cholesky(precision)
    differences = means - projected_mean.unsqueeze(0)
    between = torch.einsum(
        "n,ni,nj->ij",
        weights,
        differences,
        differences,
    )
    projected_covariance = shared_covariance + between
    projected_cholesky = torch.linalg.cholesky(projected_covariance)
    sample = projected_mean + projected_cholesky @ noise
    logdet_projected = 2.0 * torch.log(torch.diagonal(projected_cholesky)).sum()
    analytic_entropy = 0.5 * (
        dimension * (1.0 + math.log(2.0 * math.pi)) + logdet_projected
    )
    solved_covariance = torch.cholesky_solve(
        shared_covariance,
        projected_cholesky,
    )
    trace = torch.trace(solved_covariance)
    solved_differences = torch.cholesky_solve(
        differences.unsqueeze(-1),
        projected_cholesky,
    ).squeeze(-1)
    mahalanobis = torch.sum(differences * solved_differences, dim=1)
    logdet_shared = -2.0 * torch.log(torch.diagonal(precision)).sum()
    component_kls = 0.5 * (
        trace + mahalanobis - dimension + logdet_projected - logdet_shared
    )
    component_kl_upper_bound = torch.sum(weights * component_kls)
    projection_identity = _identity(
        "vfe4.h6.moment-projection-evaluation.v3",
        {
            "receiver_t": first.receiver_t,
            "component_identities": [
                component.component_identity_sha256 for component in terminal_components
            ],
            "weights": _tensor_fingerprint(weights),
            "component_means": _tensor_fingerprint(means),
            "shared_covariance": _tensor_fingerprint(shared_covariance),
            "projected_mean": _tensor_fingerprint(projected_mean),
            "projected_covariance": _tensor_fingerprint(projected_covariance),
            "component_kl_upper_bound": _tensor_fingerprint(component_kl_upper_bound),
            "bound_label": "weighted_component_kl_upper_bound",
        },
    )
    return MomentProjectionEvaluationV3(
        component_weights=weights,
        component_means=means,
        shared_component_covariance=shared_covariance,
        projected_mean=projected_mean,
        projected_covariance=projected_covariance,
        projected_cholesky=projected_cholesky,
        sample=sample,
        analytic_entropy=analytic_entropy,
        component_kl_upper_bound=component_kl_upper_bound,
        projection_identity_sha256=projection_identity,
    )


def _split_channels(
    value: Tensor,
    *,
    latent_width: int,
    model_channel_enabled: bool,
) -> tuple[Tensor, Tensor | None]:
    state = value[:latent_width]
    model = value[latent_width : 2 * latent_width] if model_channel_enabled else None
    return state, model


def _diagonal_gaussian_log_prob(
    value: Tensor,
    mean: Tensor,
    log_scale: Tensor,
) -> Tensor:
    standardized = (value - mean) * torch.exp(-log_scale)
    return -0.5 * torch.sum(
        standardized.square() + 2.0 * log_scale + math.log(2.0 * math.pi)
    )


def _prefix(
    *,
    receiver_t: int,
    observed_tokens: Tensor,
    model: "LatentLanguageArmModel",
) -> CausalPrefix:
    return CausalPrefix.create(
        receiver_t=receiver_t,
        vocabulary=model.vocabulary,
        token_ids=observed_tokens[: receiver_t - 1].contiguous(),
    )


def _categorical_row(
    bank: CategoricalSourceBank | AbsentSourceBank,
    *,
    receiver_t: int,
) -> CategoricalSourceRow | None:
    if type(bank) is AbsentSourceBank:
        return None
    if type(bank) is not CategoricalSourceBank:
        raise ValueError("trajectory source bank type is invalid")
    return bank.rows[receiver_t - 1]


def _generative_source_log_prior(
    *,
    model: "LatentLanguageArmModel",
    bank: SourceBankName,
    row: CategoricalSourceRow,
    prefix: CausalPrefix,
    sampled_history: Tensor,
) -> Tensor:
    if model.source_mode != "categorical":
        raise ValueError("categorical trajectory rows require a categorical model")
    if model.prior_variant == "fixed":
        full = (
            model.state_source_log_probs(row.receiver_t)
            if bank == "state"
            else model.model_source_log_probs(row.receiver_t)
        )
    else:
        full = (
            model.state_source_log_probs(
                row.receiver_t,
                prefix=prefix,
                earlier_latents=sampled_history,
            )
            if bank == "state"
            else model.model_source_log_probs(
                row.receiver_t,
                prefix=prefix,
                earlier_latents=sampled_history,
            )
        )
    indices = torch.tensor(
        row.support,
        dtype=torch.int64,
        device=full.device,
    )
    if full.ndim != 1 or full.numel() < row.receiver_t:
        raise ValueError("live generative source prior has an invalid support row")
    selected = full.index_select(0, indices)
    return _require_float64(
        selected,
        name="live sampled generative source prior",
        shape=(len(row.support),),
    )


def _state_transition_log_prob(
    *,
    model: "LatentLanguageArmModel",
    receiver_t: int,
    source_j: int,
    sampled_history: tuple[Tensor, ...],
    current_value: Tensor,
    frame_cache: object,
) -> Tensor:
    current_state, current_model = _split_channels(
        current_value,
        latent_width=model.latent_width,
        model_channel_enabled=model.model_channel_enabled,
    )
    source_state, _ = _split_channels(
        sampled_history[source_j],
        latent_width=model.latent_width,
        model_channel_enabled=model.model_channel_enabled,
    )
    mean = model.state_transition_mean(
        receiver_t=receiver_t,
        source_j=source_j,
        source_state=source_state,
        current_model=current_model,
        frame_cache=frame_cache,  # type: ignore[arg-type]
    )
    return _diagonal_gaussian_log_prob(
        current_state,
        mean,
        model.state_transition_log_scale,
    )


def _model_transition_log_prob(
    *,
    model: "LatentLanguageArmModel",
    receiver_t: int,
    source_j: int,
    sampled_history: tuple[Tensor, ...],
    current_value: Tensor,
    frame_cache: object,
) -> Tensor:
    _current_state, current_model = _split_channels(
        current_value,
        latent_width=model.latent_width,
        model_channel_enabled=model.model_channel_enabled,
    )
    if current_model is None:
        raise ValueError("model transition requested for an absent channel")
    _source_state, source_model = _split_channels(
        sampled_history[source_j],
        latent_width=model.latent_width,
        model_channel_enabled=model.model_channel_enabled,
    )
    if source_model is None:
        raise ValueError("model transition history is structurally absent")
    mean = model.model_transition_mean(
        receiver_t=receiver_t,
        source_j=source_j,
        source_model=source_model,
        frame_cache=frame_cache,  # type: ignore[arg-type]
    )
    return _diagonal_gaussian_log_prob(
        current_model,
        mean,
        model.model_transition_log_scale,
    )


def _emission_log_prob(
    *,
    model: "LatentLanguageArmModel",
    current_value: Tensor,
    observed_token_id: int,
) -> Tensor:
    state, model_value = _split_channels(
        current_value,
        latent_width=model.latent_width,
        model_channel_enabled=model.model_channel_enabled,
    )
    return model.emission_log_probs(
        state=state,
        model=model_value,
    )[observed_token_id]


def _source_row_record(
    *,
    bank: SourceBankName,
    row: CategoricalSourceRow,
    generative_log_prior: Tensor,
    transition_log_probabilities: Tensor,
    sampled_earlier_latents: Tensor,
) -> H6SourceRowEvaluationV3:
    posterior = row.log_probabilities
    probabilities = posterior.exp()
    categorical_entropy = -(probabilities * posterior).sum()
    if not bool(
        torch.allclose(
            categorical_entropy,
            row.entropy,
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        raise ValueError("categorical entropy must occur exactly once")
    source_contribution = torch.sum(probabilities * generative_log_prior)
    transition_contribution = torch.sum(probabilities * transition_log_probabilities)
    entropy_contribution = categorical_entropy
    combined = source_contribution + transition_contribution + entropy_contribution
    identity = _identity(
        "vfe4.h6.source-row-evaluation.v3",
        {
            "bank": bank,
            "receiver_t": row.receiver_t,
            "support": list(row.support),
            "recognition_row_identity": row.row_identity_sha256,
            "posterior": _tensor_fingerprint(posterior),
            "generative_prior": _tensor_fingerprint(generative_log_prior),
            "transition": _tensor_fingerprint(transition_log_probabilities),
            "sampled_history": _tensor_fingerprint(sampled_earlier_latents),
        },
    )
    return H6SourceRowEvaluationV3(
        bank=bank,
        receiver_t=row.receiver_t,
        support=row.support,
        posterior_log_probabilities=posterior,
        generative_log_prior=generative_log_prior,
        transition_log_probabilities=transition_log_probabilities,
        sampled_earlier_latents=sampled_earlier_latents,
        categorical_entropy=categorical_entropy,
        source_log_prior_contribution=source_contribution,
        transition_contribution=transition_contribution,
        entropy_contribution=entropy_contribution,
        combined_reduction=combined,
        row_evaluation_identity_sha256=identity,
    )


def _term(
    partition: H6ObjectivePartitionV3,
    receiver_t: int,
    value: Tensor,
) -> H6ObjectiveTermV3:
    identity = _identity(
        "vfe4.h6.objective-term.v3",
        {
            "partition": partition,
            "receiver_t": receiver_t,
            "value": _tensor_fingerprint(value),
        },
    )
    return H6ObjectiveTermV3(
        partition,
        receiver_t,
        value,
        identity,
    )


def _trajectory_live_tensors(
    trajectory: LanguageRecognitionTrajectory,
) -> tuple[Tensor, ...]:
    values: list[Tensor] = [
        trajectory.contexts,
        trajectory.base_means,
        trajectory.shared_precision_cholesky,
    ]
    for bank in (trajectory.state_source, trajectory.model_source):
        if type(bank) is CategoricalSourceBank:
            for row in bank.rows:
                values.extend(
                    (
                        row.log_prior_baseline,
                        row.residual_scores,
                        row.log_probabilities,
                        row.entropy,
                    )
                )
    for components in trajectory.receiver_components:
        for component in components:
            values.extend(
                (
                    component.mean,
                    component.precision_cholesky,
                    component.log_probability,
                )
            )
    return tuple(values)


def _validate_active_block(
    *,
    model: "LatentLanguageArmModel",
    trajectory: LanguageRecognitionTrajectory,
    active_parameter_block: H6ActiveParameterBlockV3,
) -> None:
    if active_parameter_block == "recognition":
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError(
                "recognition-phase ELBO requires every model parameter frozen"
            )
        return
    if active_parameter_block != "model":
        raise ValueError("active_parameter_block must be recognition or model")
    if any(value.requires_grad for value in _trajectory_live_tensors(trajectory)):
        raise ValueError(
            "model-phase ELBO requires an immutable detached recognition trajectory"
        )


def _sum_terms(terms: tuple[H6ObjectiveTermV3, ...]) -> Tensor:
    total = terms[0].value
    for term in terms[1:]:
        total = total + term.value
    return total


def evaluate_h6_prediction_elbo_v3(
    *,
    model: "LatentLanguageArmModel",
    trajectory: LanguageRecognitionTrajectory,
    observed_tokens: Tensor,
    base_noise: Tensor,
    mixture_mode: H6MixtureModeV3,
    active_parameter_block: H6ActiveParameterBlockV3,
) -> H6PredictionObjectiveEstimateV3:
    """Evaluate one exact-source, one-sample positive language ELBO."""

    # Local import avoids the historical arms -> objective package import cycle.
    from vfe4.predictive.identities import canonical_model_state_sha256
    from vfe4.training.arms import LatentLanguageArmModel

    if type(model) is not LatentLanguageArmModel:
        raise ValueError("model must be an exact LatentLanguageArmModel")
    if type(trajectory) is not LanguageRecognitionTrajectory:
        raise ValueError("trajectory must be an exact LanguageRecognitionTrajectory")
    trajectory.__post_init__()
    if mixture_mode not in ("exact", "moment_projection"):
        raise ValueError("mixture_mode must be exact or moment_projection")
    horizon = model.horizon
    dimension = model.latent_width * (2 if model.model_channel_enabled else 1)
    tokens = observed_tokens
    if (
        type(tokens) is not Tensor
        or tokens.device.type != "cpu"
        or tokens.dtype is not torch.int64
        or tokens.shape != (horizon,)
        or not tokens.is_contiguous()
        or bool(torch.any(tokens < 0))
        or bool(torch.any(tokens >= model.vocabulary.size))
    ):
        raise ValueError(
            "observed_tokens must be contiguous CPU int64 shape (horizon,)"
        )
    if (
        trajectory.conditioning.horizon != horizon
        or trajectory.base_means.shape != (horizon + 1, dimension)
        or trajectory.conditioning.observed_tokens.value().device.type != "cpu"
        or not torch.equal(
            trajectory.conditioning.observed_tokens.value(),
            observed_tokens,
        )
    ):
        raise ValueError("recognition trajectory does not match the model/example")
    noise = _require_float64(
        base_noise,
        name="base_noise",
        shape=(horizon + 1, dimension),
    )
    if noise.device != trajectory.base_means.device or noise.requires_grad:
        raise ValueError(
            "base_noise must be graph-free on the recognition trajectory device"
        )
    if next(model.parameters()).device != trajectory.base_means.device:
        raise ValueError("model and recognition trajectory devices must match")
    source_model_state_sha256 = canonical_model_state_sha256(model)
    if trajectory.source_model_state_sha256 != source_model_state_sha256:
        raise ValueError(
            "recognition trajectory source-model state does not match "
            "the live generative model"
        )
    _validate_active_block(
        model=model,
        trajectory=trajectory,
        active_parameter_block=active_parameter_block,
    )

    precision = trajectory.shared_precision_cholesky
    gaussian_entropy = _precision_gaussian_entropy(precision)
    source_independent_samples = tuple(
        _sample_precision_gaussian(
            trajectory.receiver_components[receiver_t][0].mean,
            precision,
            noise[receiver_t],
        )
        for receiver_t in range(horizon)
    )
    state_histories = tuple(
        _split_channels(
            value,
            latent_width=model.latent_width,
            model_channel_enabled=model.model_channel_enabled,
        )[0]
        for value in source_independent_samples
    )
    model_histories = (
        tuple(
            _split_channels(
                value,
                latent_width=model.latent_width,
                model_channel_enabled=True,
            )[1]
            for value in source_independent_samples
        )
        if model.model_channel_enabled
        else ()
    )
    if model.model_channel_enabled and any(value is None for value in model_histories):
        raise RuntimeError("model-channel sampled history is incomplete")

    initial_state, initial_model = _split_channels(
        source_independent_samples[0],
        latent_width=model.latent_width,
        model_channel_enabled=model.model_channel_enabled,
    )
    initial = model.initial_log_prob(
        state=initial_state,
        model=initial_model,
    )
    terms: list[H6ObjectiveTermV3] = [
        _term("initial", 0, initial),
        _term("gaussian_entropy", 0, gaussian_entropy),
    ]
    source_records: list[H6SourceRowEvaluationV3] = []
    independent_receiver_totals: list[Tensor] = []

    frame_scope = (
        model.shared_frame_evaluation()
        if model.map_mode == "shared_vertex_coboundary"
        else nullcontext(None)
    )
    with frame_scope as frame_cache:
        for receiver_t in range(1, horizon):
            current = source_independent_samples[receiver_t]
            receiver_terms: list[Tensor] = []
            prefix = _prefix(
                receiver_t=receiver_t,
                observed_tokens=observed_tokens,
                model=model,
            )
            state_row = _categorical_row(
                trajectory.state_source,
                receiver_t=receiver_t,
            )
            if state_row is not None:
                sampled_state_history = torch.stack(state_histories[:receiver_t])
                state_prior = _generative_source_log_prior(
                    model=model,
                    bank="state",
                    row=state_row,
                    prefix=prefix,
                    sampled_history=sampled_state_history,
                )
                state_transitions = torch.stack(
                    tuple(
                        _state_transition_log_prob(
                            model=model,
                            receiver_t=receiver_t,
                            source_j=source_j,
                            sampled_history=source_independent_samples,
                            current_value=current,
                            frame_cache=frame_cache,
                        )
                        for source_j in state_row.support
                    )
                )
                state_record = _source_row_record(
                    bank="state",
                    row=state_row,
                    generative_log_prior=state_prior,
                    transition_log_probabilities=state_transitions,
                    sampled_earlier_latents=sampled_state_history,
                )
                source_records.append(state_record)
                terms.extend(
                    (
                        _term(
                            "state_source",
                            receiver_t,
                            state_record.source_log_prior_contribution
                            + state_record.entropy_contribution,
                        ),
                        _term(
                            "state_transition",
                            receiver_t,
                            state_record.transition_contribution,
                        ),
                    )
                )
                receiver_terms.append(state_record.combined_reduction)
            else:
                state_transition = _state_transition_log_prob(
                    model=model,
                    receiver_t=receiver_t,
                    source_j=receiver_t - 1,
                    sampled_history=source_independent_samples,
                    current_value=current,
                    frame_cache=frame_cache,
                )
                terms.append(_term("state_transition", receiver_t, state_transition))
                receiver_terms.append(state_transition)

            if model.model_channel_enabled:
                model_row = _categorical_row(
                    trajectory.model_source,
                    receiver_t=receiver_t,
                )
                if model_row is not None:
                    sampled_model_history = torch.stack(
                        tuple(
                            value
                            for value in model_histories[:receiver_t]
                            if value is not None
                        )
                    )
                    model_prior = _generative_source_log_prior(
                        model=model,
                        bank="model",
                        row=model_row,
                        prefix=prefix,
                        sampled_history=sampled_model_history,
                    )
                    model_transitions = torch.stack(
                        tuple(
                            _model_transition_log_prob(
                                model=model,
                                receiver_t=receiver_t,
                                source_j=source_j,
                                sampled_history=source_independent_samples,
                                current_value=current,
                                frame_cache=frame_cache,
                            )
                            for source_j in model_row.support
                        )
                    )
                    model_record = _source_row_record(
                        bank="model",
                        row=model_row,
                        generative_log_prior=model_prior,
                        transition_log_probabilities=model_transitions,
                        sampled_earlier_latents=sampled_model_history,
                    )
                    source_records.append(model_record)
                    terms.extend(
                        (
                            _term(
                                "model_source",
                                receiver_t,
                                model_record.source_log_prior_contribution
                                + model_record.entropy_contribution,
                            ),
                            _term(
                                "model_transition",
                                receiver_t,
                                model_record.transition_contribution,
                            ),
                        )
                    )
                    receiver_terms.append(model_record.combined_reduction)
                else:
                    model_transition = _model_transition_log_prob(
                        model=model,
                        receiver_t=receiver_t,
                        source_j=receiver_t - 1,
                        sampled_history=source_independent_samples,
                        current_value=current,
                        frame_cache=frame_cache,
                    )
                    terms.append(
                        _term(
                            "model_transition",
                            receiver_t,
                            model_transition,
                        )
                    )
                    receiver_terms.append(model_transition)

            emission = _emission_log_prob(
                model=model,
                current_value=current,
                observed_token_id=int(observed_tokens[receiver_t - 1].item()),
            )
            terms.extend(
                (
                    _term("emission", receiver_t, emission),
                    _term(
                        "gaussian_entropy",
                        receiver_t,
                        gaussian_entropy,
                    ),
                )
            )
            receiver_terms.extend((emission, gaussian_entropy))
            receiver_total = receiver_terms[0]
            for value in receiver_terms[1:]:
                receiver_total = receiver_total + value
            independent_receiver_totals.append(receiver_total)

        terminal_t = horizon
        terminal_prefix = _prefix(
            receiver_t=terminal_t,
            observed_tokens=observed_tokens,
            model=model,
        )
        state_row = _categorical_row(
            trajectory.state_source,
            receiver_t=terminal_t,
        )
        model_row = (
            _categorical_row(
                trajectory.model_source,
                receiver_t=terminal_t,
            )
            if model.model_channel_enabled
            else None
        )
        sampled_state_history = torch.stack(state_histories)
        sampled_model_history = (
            torch.stack(tuple(value for value in model_histories if value is not None))
            if model.model_channel_enabled
            else None
        )
        state_prior = (
            _generative_source_log_prior(
                model=model,
                bank="state",
                row=state_row,
                prefix=terminal_prefix,
                sampled_history=sampled_state_history,
            )
            if state_row is not None
            else None
        )
        model_prior = (
            _generative_source_log_prior(
                model=model,
                bank="model",
                row=model_row,
                prefix=terminal_prefix,
                sampled_history=sampled_model_history,
            )
            if model_row is not None and sampled_model_history is not None
            else None
        )

        terminal_component_records: list[H6TerminalComponentEvaluationV3] = []
        if mixture_mode == "exact":
            for component in trajectory.terminal_components:
                sample = _sample_precision_gaussian(
                    component.mean,
                    precision,
                    noise[terminal_t],
                )
                state_source_j = (
                    component.state_source_j
                    if state_row is not None
                    else terminal_t - 1
                )
                if state_source_j is None:
                    raise ValueError("terminal state source label is missing")
                state_transition = _state_transition_log_prob(
                    model=model,
                    receiver_t=terminal_t,
                    source_j=state_source_j,
                    sampled_history=source_independent_samples,
                    current_value=sample,
                    frame_cache=frame_cache,
                )
                state_log_prior = sample.new_zeros(())
                state_log_q = sample.new_zeros(())
                if state_row is not None:
                    state_index = state_row.support.index(state_source_j)
                    assert state_prior is not None
                    state_log_prior = state_prior[state_index]
                    state_log_q = state_row.log_probabilities[state_index]

                model_source_j: int | None = None
                model_log_prior = sample.new_zeros(())
                model_log_q = sample.new_zeros(())
                model_transition = sample.new_zeros(())
                if model.model_channel_enabled:
                    model_source_j = (
                        component.model_source_j
                        if model_row is not None
                        else terminal_t - 1
                    )
                    if model_source_j is None:
                        raise ValueError("terminal model source label is missing")
                    model_transition = _model_transition_log_prob(
                        model=model,
                        receiver_t=terminal_t,
                        source_j=model_source_j,
                        sampled_history=source_independent_samples,
                        current_value=sample,
                        frame_cache=frame_cache,
                    )
                    if model_row is not None:
                        model_index = model_row.support.index(model_source_j)
                        assert model_prior is not None
                        model_log_prior = model_prior[model_index]
                        model_log_q = model_row.log_probabilities[model_index]

                emission = _emission_log_prob(
                    model=model,
                    current_value=sample,
                    observed_token_id=int(observed_tokens[terminal_t - 1].item()),
                )
                categorical_log_ratio = (
                    state_log_prior - state_log_q + model_log_prior - model_log_q
                )
                bracket = (
                    categorical_log_ratio
                    + state_transition
                    + model_transition
                    + emission
                )
                weighted = component.probability * bracket
                component_identity = _identity(
                    "vfe4.h6.terminal-component-evaluation.v3",
                    {
                        "recognition_component_identity": (
                            component.component_identity_sha256
                        ),
                        "state_source_j": state_source_j,
                        "model_source_j": model_source_j,
                        "sample": _tensor_fingerprint(sample),
                        "state_log_prior": _tensor_fingerprint(state_log_prior),
                        "state_transition": _tensor_fingerprint(state_transition),
                        "model_log_prior": _tensor_fingerprint(model_log_prior),
                        "model_transition": _tensor_fingerprint(model_transition),
                        "emission": _tensor_fingerprint(emission),
                    },
                )
                terminal_component_records.append(
                    H6TerminalComponentEvaluationV3(
                        state_source_j=(
                            component.state_source_j if state_row is not None else None
                        ),
                        model_source_j=(
                            component.model_source_j if model_row is not None else None
                        ),
                        posterior_log_weight=component.log_probability,
                        sample=sample,
                        state_log_prior=state_log_prior,
                        state_transition_log_prob=state_transition,
                        model_log_prior=model_log_prior,
                        model_transition_log_prob=model_transition,
                        emission_log_prob=emission,
                        categorical_log_ratio=categorical_log_ratio,
                        bracket=bracket,
                        weighted_contribution=weighted,
                        component_evaluation_identity_sha256=(component_identity),
                    )
                )
            terminal_joint = terminal_component_records[0].weighted_contribution
            for record in terminal_component_records[1:]:
                terminal_joint = terminal_joint + record.weighted_contribution
            source_law: H6EvaluatedRecognitionLawV3
            exact_identity = _identity(
                "vfe4.h6.exact-source-mixture-evaluation.v3",
                {
                    "trajectory_identity": (trajectory.trajectory_identity_sha256),
                    "source_independent_samples": [
                        _tensor_fingerprint(value)
                        for value in source_independent_samples
                    ],
                    "terminal_component_evaluations": [
                        record.component_evaluation_identity_sha256
                        for record in terminal_component_records
                    ],
                },
            )
            # Source rows are appended below after terminal conditional
            # transition reductions are materialized.
            source_law = ExactSourceMixtureEvaluationV3(
                trajectory_identity_sha256=(trajectory.trajectory_identity_sha256),
                source_independent_samples=source_independent_samples,
                source_rows=(),
                terminal_components=tuple(terminal_component_records),
                shared_precision_cholesky=precision,
                component_gaussian_entropy=gaussian_entropy,
                law_identity_sha256=exact_identity,
            )
            terminal_value_for_transition = None
            terminal_entropy = gaussian_entropy
        else:
            projection = project_terminal_mixture_v3(
                terminal_components=trajectory.terminal_components,
                base_noise=noise[terminal_t],
            )
            source_law = projection
            terminal_value_for_transition = projection.sample
            terminal_entropy = projection.analytic_entropy

        terminal_state_record: H6SourceRowEvaluationV3 | None = None
        if state_row is not None:
            assert state_prior is not None
            if mixture_mode == "exact":
                state_transition_values: list[Tensor] = []
                for source_j, state_log_q in zip(
                    state_row.support,
                    state_row.log_probabilities.unbind(),
                    strict=True,
                ):
                    weighted = state_log_q.new_zeros(())
                    for record in terminal_component_records:
                        if record.state_source_j == source_j:
                            weighted = (
                                weighted
                                + record.posterior_log_weight.exp()
                                * record.state_transition_log_prob
                            )
                    state_transition_values.append(weighted / state_log_q.exp())
                terminal_state_transitions = torch.stack(tuple(state_transition_values))
            else:
                assert terminal_value_for_transition is not None
                terminal_state_transitions = torch.stack(
                    tuple(
                        _state_transition_log_prob(
                            model=model,
                            receiver_t=terminal_t,
                            source_j=source_j,
                            sampled_history=source_independent_samples,
                            current_value=terminal_value_for_transition,
                            frame_cache=frame_cache,
                        )
                        for source_j in state_row.support
                    )
                )
            terminal_state_record = _source_row_record(
                bank="state",
                row=state_row,
                generative_log_prior=state_prior,
                transition_log_probabilities=terminal_state_transitions,
                sampled_earlier_latents=sampled_state_history,
            )
            source_records.append(terminal_state_record)

        terminal_model_record: H6SourceRowEvaluationV3 | None = None
        if model_row is not None:
            assert model_prior is not None
            assert sampled_model_history is not None
            if mixture_mode == "exact":
                model_transition_values: list[Tensor] = []
                for source_j, model_log_q in zip(
                    model_row.support,
                    model_row.log_probabilities.unbind(),
                    strict=True,
                ):
                    weighted = model_log_q.new_zeros(())
                    for record in terminal_component_records:
                        if record.model_source_j == source_j:
                            weighted = (
                                weighted
                                + record.posterior_log_weight.exp()
                                * record.model_transition_log_prob
                            )
                    model_transition_values.append(weighted / model_log_q.exp())
                terminal_model_transitions = torch.stack(tuple(model_transition_values))
            else:
                assert terminal_value_for_transition is not None
                terminal_model_transitions = torch.stack(
                    tuple(
                        _model_transition_log_prob(
                            model=model,
                            receiver_t=terminal_t,
                            source_j=source_j,
                            sampled_history=source_independent_samples,
                            current_value=terminal_value_for_transition,
                            frame_cache=frame_cache,
                        )
                        for source_j in model_row.support
                    )
                )
            terminal_model_record = _source_row_record(
                bank="model",
                row=model_row,
                generative_log_prior=model_prior,
                transition_log_probabilities=terminal_model_transitions,
                sampled_earlier_latents=sampled_model_history,
            )
            source_records.append(terminal_model_record)

        if mixture_mode == "moment_projection":
            assert terminal_value_for_transition is not None
            terminal_parts: list[Tensor] = []
            if terminal_state_record is not None:
                terminal_parts.append(terminal_state_record.combined_reduction)
            else:
                state_transition = _state_transition_log_prob(
                    model=model,
                    receiver_t=terminal_t,
                    source_j=terminal_t - 1,
                    sampled_history=source_independent_samples,
                    current_value=terminal_value_for_transition,
                    frame_cache=frame_cache,
                )
                terminal_parts.append(state_transition)
            if model.model_channel_enabled:
                if terminal_model_record is not None:
                    terminal_parts.append(terminal_model_record.combined_reduction)
                else:
                    model_transition = _model_transition_log_prob(
                        model=model,
                        receiver_t=terminal_t,
                        source_j=terminal_t - 1,
                        sampled_history=source_independent_samples,
                        current_value=terminal_value_for_transition,
                        frame_cache=frame_cache,
                    )
                    terminal_parts.append(model_transition)
            projected_emission = _emission_log_prob(
                model=model,
                current_value=terminal_value_for_transition,
                observed_token_id=int(observed_tokens[terminal_t - 1].item()),
            )
            terminal_parts.append(projected_emission)
            terminal_joint = terminal_parts[0]
            for value in terminal_parts[1:]:
                terminal_joint = terminal_joint + value

        terminal_decomposition: list[Tensor] = []
        if terminal_state_record is not None:
            state_source_term = (
                terminal_state_record.source_log_prior_contribution
                + terminal_state_record.entropy_contribution
            )
            terms.extend(
                (
                    _term(
                        "state_source",
                        terminal_t,
                        state_source_term,
                    ),
                    _term(
                        "state_transition",
                        terminal_t,
                        terminal_state_record.transition_contribution,
                    ),
                )
            )
            terminal_decomposition.extend(
                (
                    state_source_term,
                    terminal_state_record.transition_contribution,
                )
            )
        else:
            if mixture_mode == "exact":
                state_transition_term = sum(
                    (
                        record.posterior_log_weight.exp()
                        * record.state_transition_log_prob
                        for record in terminal_component_records
                    ),
                    start=precision.new_zeros(()),
                )
            else:
                assert terminal_value_for_transition is not None
                state_transition_term = _state_transition_log_prob(
                    model=model,
                    receiver_t=terminal_t,
                    source_j=terminal_t - 1,
                    sampled_history=source_independent_samples,
                    current_value=terminal_value_for_transition,
                    frame_cache=frame_cache,
                )
            terms.append(
                _term(
                    "state_transition",
                    terminal_t,
                    state_transition_term,
                )
            )
            terminal_decomposition.append(state_transition_term)

        if model.model_channel_enabled:
            if terminal_model_record is not None:
                model_source_term = (
                    terminal_model_record.source_log_prior_contribution
                    + terminal_model_record.entropy_contribution
                )
                terms.extend(
                    (
                        _term(
                            "model_source",
                            terminal_t,
                            model_source_term,
                        ),
                        _term(
                            "model_transition",
                            terminal_t,
                            terminal_model_record.transition_contribution,
                        ),
                    )
                )
                terminal_decomposition.extend(
                    (
                        model_source_term,
                        terminal_model_record.transition_contribution,
                    )
                )
            else:
                if mixture_mode == "exact":
                    model_transition_term = sum(
                        (
                            record.posterior_log_weight.exp()
                            * record.model_transition_log_prob
                            for record in terminal_component_records
                        ),
                        start=precision.new_zeros(()),
                    )
                else:
                    assert terminal_value_for_transition is not None
                    model_transition_term = _model_transition_log_prob(
                        model=model,
                        receiver_t=terminal_t,
                        source_j=terminal_t - 1,
                        sampled_history=source_independent_samples,
                        current_value=terminal_value_for_transition,
                        frame_cache=frame_cache,
                    )
                terms.append(
                    _term(
                        "model_transition",
                        terminal_t,
                        model_transition_term,
                    )
                )
                terminal_decomposition.append(model_transition_term)

        if mixture_mode == "exact":
            terminal_emission = sum(
                (
                    record.posterior_log_weight.exp() * record.emission_log_prob
                    for record in terminal_component_records
                ),
                start=precision.new_zeros(()),
            )
        else:
            assert terminal_value_for_transition is not None
            terminal_emission = _emission_log_prob(
                model=model,
                current_value=terminal_value_for_transition,
                observed_token_id=int(observed_tokens[terminal_t - 1].item()),
            )
        terms.extend(
            (
                _term("emission", terminal_t, terminal_emission),
                _term(
                    "gaussian_entropy",
                    terminal_t,
                    terminal_entropy,
                ),
            )
        )
        terminal_decomposition.append(terminal_emission)
        terminal_decomposed_joint = terminal_decomposition[0]
        for value in terminal_decomposition[1:]:
            terminal_decomposed_joint = terminal_decomposed_joint + value
        if not bool(
            torch.allclose(
                terminal_decomposed_joint,
                terminal_joint,
                rtol=1e-11,
                atol=1e-11,
            )
        ):
            raise RuntimeError(
                "terminal source/transition/emission partition does not "
                "equal the explicit joint source reduction"
            )

    if type(source_law) is ExactSourceMixtureEvaluationV3:
        completed_exact_identity = _identity(
            "vfe4.h6.exact-source-mixture-evaluation.v3",
            {
                "trajectory_identity": (source_law.trajectory_identity_sha256),
                "source_independent_samples": [
                    _tensor_fingerprint(value)
                    for value in source_law.source_independent_samples
                ],
                "source_row_evaluations": [
                    record.row_evaluation_identity_sha256 for record in source_records
                ],
                "terminal_component_evaluations": [
                    record.component_evaluation_identity_sha256
                    for record in source_law.terminal_components
                ],
            },
        )
        source_law = ExactSourceMixtureEvaluationV3(
            trajectory_identity_sha256=source_law.trajectory_identity_sha256,
            source_independent_samples=source_law.source_independent_samples,
            source_rows=tuple(source_records),
            terminal_components=source_law.terminal_components,
            shared_precision_cholesky=source_law.shared_precision_cholesky,
            component_gaussian_entropy=source_law.component_gaussian_entropy,
            law_identity_sha256=completed_exact_identity,
        )

    ordered_terms = tuple(terms)
    canonical_total = _sum_terms(ordered_terms)
    independent_total = initial + gaussian_entropy
    for receiver_total in independent_receiver_totals:
        independent_total = independent_total + receiver_total
    independent_total = independent_total + terminal_joint + terminal_entropy
    if not bool(
        torch.allclose(
            canonical_total,
            independent_total,
            rtol=1e-11,
            atol=1e-11,
        )
    ):
        raise RuntimeError(
            "canonical ordered ELBO does not equal its independent accumulation"
        )
    # The independent check above closes bookkeeping; downstream autograd gets
    # one and only one live scalar graph.
    exposed_total = canonical_total
    loss = -exposed_total
    estimate_identity = _identity(
        "vfe4.h6.prediction-objective-estimate.v3",
        {
            "mixture_mode": mixture_mode,
            "active_parameter_block": active_parameter_block,
            "trajectory_identity": trajectory.trajectory_identity_sha256,
            "source_law_identity": (
                source_law.law_identity_sha256
                if type(source_law) is ExactSourceMixtureEvaluationV3
                else source_law.projection_identity_sha256
            ),
            "ordered_terms": [term.term_identity_sha256 for term in ordered_terms],
            "terminal_joint": _tensor_fingerprint(terminal_joint),
            "elbo": _tensor_fingerprint(exposed_total),
        },
    )
    if canonical_model_state_sha256(model) != source_model_state_sha256:
        raise ValueError("live generative model changed during objective evaluation")
    return H6PredictionObjectiveEstimateV3(
        mixture_mode=mixture_mode,
        active_parameter_block=active_parameter_block,
        source_law=source_law,
        evaluated_source_rows=tuple(source_records),
        ordered_terms=ordered_terms,
        terminal_joint_contribution=terminal_joint,
        canonical_ordered_total=exposed_total,
        independently_accumulated_total=exposed_total,
        elbo=exposed_total,
        loss=loss,
        estimate_identity_sha256=estimate_identity,
    )


__all__ = [
    "ExactSourceMixtureEvaluationV3",
    "H6ActiveParameterBlockV3",
    "H6EvaluatedRecognitionLawV3",
    "H6MixtureModeV3",
    "H6ObjectivePartitionV3",
    "H6ObjectiveTermV3",
    "H6PredictionObjectiveEstimateV3",
    "H6SourceRowEvaluationV3",
    "H6TerminalComponentEvaluationV3",
    "MomentProjectionEvaluationV3",
    "evaluate_h6_prediction_elbo_v3",
    "project_terminal_mixture_v3",
]
