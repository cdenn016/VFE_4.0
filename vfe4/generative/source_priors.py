"""Normalized causal source priors for the H6 language model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from vfe4.data.windows import CausalPrefix
from vfe4.numerics.categorical import masked_log_softmax_from_parents
from vfe4.types.h6 import (
    FrozenTensorSnapshot,
    H6LanguageStructure,
    VocabularyIdentity,
    canonical_json_bytes,
)


PriorVariant = Literal[
    "fixed",
    "parent_specific_pooled_prefix",
    "pooled_history_conditioned",
]
SourceBank = Literal["state", "model"]


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _vocabulary_sha256(vocabulary: VocabularyIdentity) -> str:
    return _owned_hash(
        "vfe4.h6.vocabulary-identity.v1",
        {
            "vocabulary_id": vocabulary.vocabulary_id,
            "size": vocabulary.size,
            "tokenizer_spec_sha256": vocabulary.tokenizer_spec_sha256,
        },
    )


def _tensor_raw_sha256(value: Tensor) -> str:
    raw = bytes(
        value.detach()
        .to(device="cpu")
        .contiguous()
        .view(torch.uint8)
        .reshape(-1)
        .tolist()
    )
    return hashlib.sha256(raw).hexdigest()


def _deterministic_matrix(rows: int, columns: int, *, scale: float) -> Tensor:
    values = torch.arange(rows * columns, dtype=torch.float64).reshape(
        rows, columns
    )
    centered = values - 0.5 * max(0, rows * columns - 1)
    return scale * centered / max(1, rows * columns)


def _snapshot_payload(snapshot: FrozenTensorSnapshot) -> dict[str, object]:
    snapshot.assert_intact()
    return {
        "dtype": snapshot.dtype,
        "shape": snapshot.shape,
        "device": snapshot.device,
        "contiguous": snapshot.contiguous,
        "requires_grad": snapshot.requires_grad,
        "storage_version": snapshot.storage_version,
        "raw_bytes_sha256": snapshot.raw_bytes_sha256,
    }


@dataclass(frozen=True)
class MaskCaseKey:
    """Complete identity of one causal source-support check."""

    fixture_sha256: str
    vocabulary_sha256: str
    predictor_config_sha256: str
    model_family_sha256: str
    prior_variant: PriorVariant
    bank: SourceBank
    receiver_t: int
    context_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "fixture_sha256",
            "vocabulary_sha256",
            "predictor_config_sha256",
            "model_family_sha256",
            "context_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if self.prior_variant not in (
            "fixed",
            "parent_specific_pooled_prefix",
            "pooled_history_conditioned",
        ):
            raise ValueError("unsupported source-prior variant")
        if self.bank not in ("state", "model"):
            raise ValueError("unsupported source bank")
        if type(self.receiver_t) is not int or self.receiver_t <= 0:
            raise ValueError("receiver_t must be a positive integer")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "fixture_sha256": self.fixture_sha256,
            "vocabulary_sha256": self.vocabulary_sha256,
            "predictor_config_sha256": self.predictor_config_sha256,
            "model_family_sha256": self.model_family_sha256,
            "prior_variant": self.prior_variant,
            "bank": self.bank,
            "receiver_t": self.receiver_t,
            "context_sha256": self.context_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return _owned_hash("vfe4.h6.mask-case-key.v1", self.canonical_payload())


@dataclass(frozen=True, slots=True)
class FixedSourceFactorContext:
    """Exact source slot proving that a fixed prior consumes no context."""

    bank: SourceBank
    receiver_t: int

    def __post_init__(self) -> None:
        if self.bank not in ("state", "model"):
            raise ValueError("unsupported fixed source-context bank")
        if type(self.receiver_t) is not int or self.receiver_t <= 0:
            raise ValueError(
                "fixed source-context receiver_t must be positive"
            )


@dataclass(frozen=True, slots=True)
class PrefixConditionedSourceFactorContext:
    """Exact live inputs needed to recompute one prefix-conditioned factor."""

    bank: SourceBank
    receiver_t: int
    prefix: CausalPrefix
    earlier_latents: Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.bank not in ("state", "model"):
            raise ValueError("unsupported prefix source-context bank")
        if type(self.receiver_t) is not int or self.receiver_t <= 0:
            raise ValueError(
                "prefix source-context receiver_t must be positive"
            )
        if type(self.prefix) is not CausalPrefix:
            raise ValueError(
                "prefix source context requires an exact CausalPrefix"
            )
        self.prefix.__post_init__()
        if self.prefix.receiver_t != self.receiver_t:
            raise ValueError(
                "prefix source-context receiver does not match its prefix"
            )
        if (
            type(self.earlier_latents) is not Tensor
            or self.earlier_latents.dtype is not torch.float64
            or self.earlier_latents.ndim != 2
            or self.earlier_latents.shape[0] != self.receiver_t
            or not bool(torch.isfinite(self.earlier_latents).all())
        ):
            raise ValueError(
                "prefix source-context earlier_latents must be finite "
                "float64 shape (receiver_t, latent_dim)"
            )


SourceFactorContext = (
    FixedSourceFactorContext | PrefixConditionedSourceFactorContext
)


@dataclass(frozen=True)
class NormalizedSourceFactor:
    """One normalized source row with immutable value and support identities."""

    mask_case_key: MaskCaseKey
    support_mask: tuple[bool, ...]
    log_probs: FrozenTensorSnapshot
    factor_identity_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        self.log_probs.assert_intact()
        return {
            "mask_case_sha256": self.mask_case_key.canonical_sha256,
            "support_mask": self.support_mask,
            "log_probs": _snapshot_payload(self.log_probs),
        }

    def __post_init__(self) -> None:
        if type(self.mask_case_key) is not MaskCaseKey:
            raise ValueError("mask_case_key must be an exact MaskCaseKey")
        if (
            type(self.support_mask) is not tuple
            or len(self.support_mask) != self.mask_case_key.receiver_t
            or not self.support_mask
            or any(type(value) is not bool for value in self.support_mask)
            or not any(self.support_mask)
        ):
            raise ValueError("support_mask must be a nonempty exact Boolean source row")
        if type(self.log_probs) is not FrozenTensorSnapshot:
            raise ValueError("log_probs must be a FrozenTensorSnapshot")
        self.log_probs.assert_intact()
        if self.log_probs.shape != (self.mask_case_key.receiver_t,):
            raise ValueError("source log-probabilities have the wrong shape")
        values = self.log_probs.value()
        if values.dtype is not torch.float64:
            raise ValueError("source log-probabilities must use float64")
        support = torch.tensor(
            self.support_mask, dtype=torch.bool, device=values.device
        )
        if not bool(torch.isfinite(values[support]).all()):
            raise ValueError("supported source log-probabilities must be finite")
        if bool(torch.any(~support)) and not bool(torch.isneginf(values[~support]).all()):
            raise ValueError("off-support source log-probabilities must be exact -inf")
        log_normalizer = torch.logsumexp(values[support], dim=0)
        allowance = 128.0 * math.ulp(1.0) * max(1, int(support.sum().item()))
        if abs(float(log_normalizer.item())) > allowance:
            raise ValueError("supported source log-probabilities must be normalized")
        _require_sha256(self.factor_identity_sha256, "factor_identity_sha256")
        expected = _owned_hash(
            "vfe4.h6.normalized-source-factor.v1",
            self.canonical_payload(),
        )
        if self.factor_identity_sha256 != expected:
            raise ValueError("source factor identity does not match the normalized record")

    @classmethod
    def create(
        cls,
        *,
        key: MaskCaseKey,
        log_probs: Tensor,
        support_mask: Tensor,
    ) -> "NormalizedSourceFactor":
        support = tuple(bool(value) for value in support_mask.tolist())
        snapshot = FrozenTensorSnapshot.capture(log_probs)
        identity = _owned_hash(
            "vfe4.h6.normalized-source-factor.v1",
            {
                "mask_case_sha256": key.canonical_sha256,
                "support_mask": support,
                "log_probs": _snapshot_payload(snapshot),
            },
        )
        return cls(
            key,
            support,
            snapshot,
            identity,
        )


class _SourcePriorBase(nn.Module):
    def __init__(
        self,
        *,
        structure: H6LanguageStructure,
        vocabulary: VocabularyIdentity,
        fixture_sha256: str,
        predictor_config_sha256: str,
        model_family_sha256: str,
    ) -> None:
        super().__init__()
        if type(structure) is not H6LanguageStructure:
            raise ValueError("structure must be an exact H6LanguageStructure")
        if type(vocabulary) is not VocabularyIdentity:
            raise ValueError("vocabulary must be an exact VocabularyIdentity")
        structure.__post_init__()
        vocabulary.__post_init__()
        self.structure = structure
        self.vocabulary = vocabulary
        self.fixture_sha256 = _require_sha256(fixture_sha256, "fixture_sha256")
        self.predictor_config_sha256 = _require_sha256(
            predictor_config_sha256, "predictor_config_sha256"
        )
        self.model_family_sha256 = _require_sha256(
            model_family_sha256, "model_family_sha256"
        )
        self.vocabulary_sha256 = _vocabulary_sha256(vocabulary)

    def _row_index(self, receiver_t: int) -> int:
        receivers = self.structure.receiver_labels
        if type(receiver_t) is not int or receiver_t not in receivers:
            raise ValueError("receiver_t is not declared by the language DAG")
        return receivers.index(receiver_t)

    def _parents(self, receiver_t: int) -> tuple[int, ...]:
        return self.structure.dag.rows[self._row_index(receiver_t)].parents

    def _key(
        self,
        *,
        prior_variant: PriorVariant,
        bank: SourceBank,
        receiver_t: int,
        context_sha256: str,
    ) -> MaskCaseKey:
        return MaskCaseKey(
            self.fixture_sha256,
            self.vocabulary_sha256,
            self.predictor_config_sha256,
            self.model_family_sha256,
            prior_variant,
            bank,
            receiver_t,
            context_sha256,
        )

    def _normalized(
        self,
        *,
        logits: Tensor,
        prior_variant: PriorVariant,
        bank: SourceBank,
        receiver_t: int,
        context_sha256: str,
    ) -> NormalizedSourceFactor:
        normalized = masked_log_softmax_from_parents(
            logits, self._parents(receiver_t), receiver_t
        )
        key = self._key(
            prior_variant=prior_variant,
            bank=bank,
            receiver_t=receiver_t,
            context_sha256=context_sha256,
        )
        return NormalizedSourceFactor.create(
            key=key,
            log_probs=normalized.log_probs,
            support_mask=normalized.support_mask,
        )


def _checked_gauge_anchored_rows(
    rows: tuple[Tensor, ...], structure: H6LanguageStructure, name: str
) -> nn.ParameterList:
    """Own supported source logits modulo one fixed-zero softmax anchor."""

    if type(rows) is not tuple or len(rows) != len(structure.dag.rows):
        raise ValueError(f"{name} must contain one row per DAG receiver")
    parameters: list[nn.Parameter] = []
    for row, dag_row in zip(rows, structure.dag.rows, strict=True):
        if (
            type(row) is not Tensor
            or row.dtype is not torch.float64
            or row.ndim != 1
            or row.shape != (dag_row.receiver_t,)
            or not bool(torch.isfinite(row).all())
        ):
            raise ValueError(
                f"{name} rows must be finite float64 vectors of receiver_t length"
            )
        free_parents = dag_row.parents[:-1]
        anchor_parent = dag_row.parents[-1]
        if not free_parents:
            continue
        canonical_free = (
            row[list(free_parents)] - row[anchor_parent]
        )
        parameters.append(nn.Parameter(canonical_free.detach().clone()))
    return nn.ParameterList(parameters)


def _free_row(
    parameters: nn.ParameterList,
    *,
    structure: H6LanguageStructure,
    receiver_t: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Tensor:
    """Return the receiver's free logits without a zero-size Parameter."""

    if type(receiver_t) is not int or receiver_t <= 0:
        raise ValueError("receiver_t must be a positive integer")
    receiver_labels = structure.receiver_labels
    if receiver_t not in receiver_labels:
        raise ValueError("receiver_t is absent from the source structure")
    row_index = receiver_labels.index(receiver_t)
    if len(structure.dag.rows[row_index].parents) == 1:
        return torch.empty(0, dtype=dtype, device=device)
    parameter_index = sum(
        len(row.parents) > 1
        for row in structure.dag.rows[:row_index]
    )
    if not 0 <= parameter_index < len(parameters):
        raise ValueError("receiver_t has no gauge-anchored free row")
    return parameters[parameter_index]


def _gauge_anchored_logits(
    *,
    free_logits: Tensor,
    receiver_t: int,
    parents: tuple[int, ...],
) -> Tensor:
    """Expand free supported logits while keeping one exact zero anchor."""

    if (
        not isinstance(free_logits, Tensor)
        or free_logits.dtype is not torch.float64
        or free_logits.ndim != 1
        or free_logits.shape != (len(parents) - 1,)
    ):
        raise ValueError("free source logits do not match the supported-parent gauge")
    logits = torch.zeros(
        receiver_t,
        dtype=free_logits.dtype,
        device=free_logits.device,
    )
    if len(parents) == 1:
        return logits
    free_indices = torch.tensor(
        parents[:-1], dtype=torch.int64, device=free_logits.device
    )
    return logits.index_copy(0, free_indices, free_logits)


class FixedSourcePrior(_SourcePriorBase):
    """Prefix-independent priors with a fixed-zero categorical gauge anchor."""

    def __init__(
        self,
        *,
        structure: H6LanguageStructure,
        vocabulary: VocabularyIdentity,
        fixture_sha256: str,
        predictor_config_sha256: str,
        model_family_sha256: str,
        state_logits: tuple[Tensor, ...],
        model_logits: tuple[Tensor, ...] | None,
    ) -> None:
        super().__init__(
            structure=structure,
            vocabulary=vocabulary,
            fixture_sha256=fixture_sha256,
            predictor_config_sha256=predictor_config_sha256,
            model_family_sha256=model_family_sha256,
        )
        self.state_source_free_logits = _checked_gauge_anchored_rows(
            state_logits, structure, "state_logits"
        )
        self.model_source_free_logits = (
            _checked_gauge_anchored_rows(
                model_logits, structure, "model_logits"
            )
            if model_logits is not None
            else None
        )

    def _context_sha256(self, receiver_t: int) -> str:
        return _owned_hash(
            "vfe4.h6.fixed-source-context.v1", {"receiver_t": receiver_t}
        )

    def state_source_log_probs(self, *, receiver_t: int) -> NormalizedSourceFactor:
        index = self._row_index(receiver_t)
        row = self.structure.dag.rows[index]
        free_logits = _free_row(
            self.state_source_free_logits,
            structure=self.structure,
            receiver_t=receiver_t,
        )
        return self._normalized(
            logits=_gauge_anchored_logits(
                free_logits=free_logits,
                receiver_t=receiver_t,
                parents=row.parents,
            ),
            prior_variant="fixed",
            bank="state",
            receiver_t=receiver_t,
            context_sha256=self._context_sha256(receiver_t),
        )

    def model_source_log_probs(self, *, receiver_t: int) -> NormalizedSourceFactor:
        if self.model_source_free_logits is None:
            raise ValueError("model source bank is structurally absent")
        index = self._row_index(receiver_t)
        row = self.structure.dag.rows[index]
        free_logits = _free_row(
            self.model_source_free_logits,
            structure=self.structure,
            receiver_t=receiver_t,
        )
        return self._normalized(
            logits=_gauge_anchored_logits(
                free_logits=free_logits,
                receiver_t=receiver_t,
                parents=row.parents,
            ),
            prior_variant="fixed",
            bank="model",
            receiver_t=receiver_t,
            context_sha256=self._context_sha256(receiver_t),
        )


class _PooledPrefixSourcePriorParameters(_SourcePriorBase):
    """Shared parameterization for the two distinct pooled-prefix scorers."""

    def __init__(
        self,
        *,
        structure: H6LanguageStructure,
        vocabulary: VocabularyIdentity,
        fixture_sha256: str,
        predictor_config_sha256: str,
        model_family_sha256: str,
        latent_dim: int,
        context_dim: int,
    ) -> None:
        super().__init__(
            structure=structure,
            vocabulary=vocabulary,
            fixture_sha256=fixture_sha256,
            predictor_config_sha256=predictor_config_sha256,
            model_family_sha256=model_family_sha256,
        )
        if type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        if type(context_dim) is not int or context_dim <= 0:
            raise ValueError("context_dim must be a positive integer")
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.token_embedding = nn.Embedding(
            vocabulary.size, context_dim, dtype=torch.float64
        )
        self.state_latent_projection = nn.Linear(
            latent_dim, context_dim, bias=False, dtype=torch.float64
        )
        self.model_latent_projection = nn.Linear(
            latent_dim, context_dim, bias=False, dtype=torch.float64
        )
        self.state_source_free_parent_keys = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        (len(row.parents) - 1, context_dim),
                        dtype=torch.float64,
                    )
                )
                for row in structure.dag.rows
                if len(row.parents) > 1
            ]
        )
        self.model_source_free_parent_keys = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        (len(row.parents) - 1, context_dim),
                        dtype=torch.float64,
                    )
                )
                for row in structure.dag.rows
                if len(row.parents) > 1
            ]
        )
        self.state_source_free_biases = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(len(row.parents) - 1, dtype=torch.float64)
                )
                for row in structure.dag.rows
                if len(row.parents) > 1
            ]
        )
        self.model_source_free_biases = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(len(row.parents) - 1, dtype=torch.float64)
                )
                for row in structure.dag.rows
                if len(row.parents) > 1
            ]
        )
        with torch.no_grad():
            self.token_embedding.weight.copy_(
                _deterministic_matrix(
                    vocabulary.size, context_dim, scale=0.125
                )
            )
            self.state_latent_projection.weight.copy_(
                _deterministic_matrix(
                    context_dim, latent_dim, scale=0.1
                )
            )
            self.model_latent_projection.weight.copy_(
                _deterministic_matrix(
                    context_dim, latent_dim, scale=0.075
                )
            )
            for row_index, (
                state_keys,
                model_keys,
                state_bias,
                model_bias,
            ) in enumerate(
                zip(
                    self.state_source_free_parent_keys,
                    self.model_source_free_parent_keys,
                    self.state_source_free_biases,
                    self.model_source_free_biases,
                    strict=True,
                )
            ):
                state_keys.copy_(
                    _deterministic_matrix(
                        state_keys.shape[0],
                        context_dim,
                        scale=0.05 * (row_index + 1),
                    )
                )
                model_keys.copy_(
                    _deterministic_matrix(
                        model_keys.shape[0],
                        context_dim,
                        scale=0.04 * (row_index + 1),
                    )
                )
                if state_bias.numel():
                    state_bias.copy_(
                        torch.linspace(
                            -0.01,
                            0.01,
                            state_bias.numel(),
                            dtype=torch.float64,
                        )
                    )
                    model_bias.copy_(
                        torch.linspace(
                            0.01,
                            -0.01,
                            model_bias.numel(),
                            dtype=torch.float64,
                        )
                    )

    def _free_parameter_index(self, receiver_t: int) -> int:
        row_index = self._row_index(receiver_t)
        if len(self.structure.dag.rows[row_index].parents) == 1:
            raise ValueError("singleton source row has no free parameter")
        return sum(
            len(row.parents) > 1
            for row in self.structure.dag.rows[:row_index]
        )

    def _checked_context(
        self, *, prefix: CausalPrefix, earlier_latents: Tensor, bank: SourceBank
    ) -> tuple[int, Tensor, str]:
        if type(prefix) is not CausalPrefix:
            raise ValueError("prefix must be an exact target-free CausalPrefix")
        prefix.__post_init__()
        if prefix.vocabulary != self.vocabulary:
            raise ValueError("CausalPrefix vocabulary does not match the source prior")
        receiver_t = prefix.receiver_t
        self._row_index(receiver_t)
        projection = (
            self.state_latent_projection
            if bank == "state"
            else self.model_latent_projection
        )
        if (
            type(earlier_latents) is not Tensor
            or earlier_latents.dtype is not torch.float64
            or earlier_latents.ndim != 2
            or earlier_latents.shape != (receiver_t, self.latent_dim)
            or earlier_latents.device != projection.weight.device
            or not bool(torch.isfinite(earlier_latents).all())
        ):
            raise ValueError(
                "earlier_latents must be finite float64 shape (receiver_t, latent_dim)"
            )
        token_ids = prefix.token_ids.to(device=self.token_embedding.weight.device)
        if token_ids.numel():
            token_context = self.token_embedding(token_ids).mean(dim=0)
        else:
            token_context = torch.zeros(
                self.context_dim,
                dtype=torch.float64,
                device=self.token_embedding.weight.device,
            )
        latent_context = projection(earlier_latents).mean(dim=0)
        context = token_context + latent_context
        context_sha256 = _owned_hash(
            self.context_hash_domain,
            {
                "prefix_sha256": prefix.prefix_sha256,
                "latent_dtype": str(earlier_latents.dtype).removeprefix("torch."),
                "latent_shape": tuple(int(size) for size in earlier_latents.shape),
                "latent_raw_sha256": _tensor_raw_sha256(earlier_latents),
            },
        )
        return receiver_t, context, context_sha256

    def state_source_log_probs(
        self, *, prefix: CausalPrefix, earlier_latents: Tensor
    ) -> NormalizedSourceFactor:
        receiver_t, context, context_sha256 = self._checked_context(
            prefix=prefix, earlier_latents=earlier_latents, bank="state"
        )
        index = self._row_index(receiver_t)
        row = self.structure.dag.rows[index]
        if len(row.parents) == 1:
            free_logits = torch.empty(
                0, dtype=context.dtype, device=context.device
            )
        else:
            free_index = self._free_parameter_index(receiver_t)
            free_logits = (
                self.state_source_free_parent_keys[free_index] @ context
                + self.state_source_free_biases[free_index]
            )
        logits = _gauge_anchored_logits(
            free_logits=free_logits,
            receiver_t=receiver_t,
            parents=row.parents,
        )
        return self._normalized(
            logits=logits,
            prior_variant="pooled_history_conditioned",
            bank="state",
            receiver_t=receiver_t,
            context_sha256=context_sha256,
        )

    def model_source_log_probs(
        self, *, prefix: CausalPrefix, earlier_latents: Tensor
    ) -> NormalizedSourceFactor:
        receiver_t, context, context_sha256 = self._checked_context(
            prefix=prefix, earlier_latents=earlier_latents, bank="model"
        )
        index = self._row_index(receiver_t)
        row = self.structure.dag.rows[index]
        if len(row.parents) == 1:
            free_logits = torch.empty(
                0, dtype=context.dtype, device=context.device
            )
        else:
            free_index = self._free_parameter_index(receiver_t)
            free_logits = (
                self.model_source_free_parent_keys[free_index] @ context
                + self.model_source_free_biases[free_index]
            )
        logits = _gauge_anchored_logits(
            free_logits=free_logits,
            receiver_t=receiver_t,
            parents=row.parents,
        )
        return self._normalized(
            logits=logits,
            prior_variant="pooled_history_conditioned",
            bank="model",
            receiver_t=receiver_t,
            context_sha256=context_sha256,
        )


class PooledHistoryConditionedSourcePrior(
    _PooledPrefixSourcePriorParameters
):
    """Legacy scorer using one pooled token-plus-latent context for every slot."""

    scorer_schema = "pooled-history-conditioned-v1"
    context_hash_domain = "vfe4.h6.pooled-history-source-context.v1"


class ParentSpecificPooledPrefixSourcePrior(
    _PooledPrefixSourcePriorParameters
):
    """Parent-specific categorical source prior with a pooled token query.

    This scorer is target blind and normalized, but its token summary is mean
    pooled. It is therefore a generative source selector, not transformer
    self-attention.
    """

    scorer_schema = "parent-specific-pooled-prefix-bilinear-v1"
    token_summary_schema = "mean-prior-token-embeddings-v1"
    parent_content_schema = "bank-projection-of-candidate-row-v1"
    anchor_schema = "last-declared-parent-complete-score-subtraction-v1"
    normalization_schema = "masked-log-softmax-from-declared-parents-v1"
    context_hash_domain = "vfe4.h6.prefix-source-context.v2"

    def _checked_parent_specific_inputs(
        self,
        *,
        prefix: CausalPrefix,
        earlier_latents: Tensor,
        bank: SourceBank,
    ) -> tuple[int, Tensor, Tensor, tuple[int, ...], str]:
        if type(prefix) is not CausalPrefix:
            raise ValueError("prefix must be an exact target-free CausalPrefix")
        prefix.__post_init__()
        if prefix.vocabulary != self.vocabulary:
            raise ValueError(
                "CausalPrefix vocabulary does not match the source prior"
            )
        receiver_t = prefix.receiver_t
        row_index = self._row_index(receiver_t)
        parents = self.structure.dag.rows[row_index].parents
        projection = (
            self.state_latent_projection
            if bank == "state"
            else self.model_latent_projection
        )
        if (
            type(earlier_latents) is not Tensor
            or earlier_latents.dtype is not torch.float64
            or earlier_latents.ndim != 2
            or earlier_latents.shape != (receiver_t, self.latent_dim)
            or earlier_latents.device != projection.weight.device
            or not bool(torch.isfinite(earlier_latents).all())
        ):
            raise ValueError(
                "earlier_latents must be finite float64 shape "
                "(receiver_t, latent_dim)"
            )
        token_ids = prefix.token_ids.to(
            device=self.token_embedding.weight.device
        )
        if token_ids.numel():
            query = self.token_embedding(token_ids).mean(dim=0)
        else:
            query = torch.zeros(
                self.context_dim,
                dtype=torch.float64,
                device=self.token_embedding.weight.device,
            )
        parent_indices = torch.tensor(
            parents, dtype=torch.int64, device=earlier_latents.device
        )
        supported_latents = earlier_latents.index_select(0, parent_indices)
        context_sha256 = _owned_hash(
            self.context_hash_domain,
            {
                "scorer_schema": self.scorer_schema,
                "token_summary": self.token_summary_schema,
                "parent_content": self.parent_content_schema,
                "anchor": self.anchor_schema,
                "normalization": self.normalization_schema,
                "prefix_sha256": prefix.prefix_sha256,
                "receiver_t": receiver_t,
                "parents": parents,
                "supported_latent_dtype": str(
                    supported_latents.dtype
                ).removeprefix("torch."),
                "supported_latent_shape": tuple(
                    int(size) for size in supported_latents.shape
                ),
                "supported_latent_raw_sha256": _tensor_raw_sha256(
                    supported_latents
                ),
            },
        )
        return (
            receiver_t,
            query,
            supported_latents,
            parents,
            context_sha256,
        )

    def _parent_specific_free_logits(
        self,
        *,
        receiver_t: int,
        query: Tensor,
        supported_latents: Tensor,
        parents: tuple[int, ...],
        bank: SourceBank,
    ) -> Tensor:
        if len(parents) == 1:
            return torch.empty(
                0, dtype=query.dtype, device=query.device
            )
        projection = (
            self.state_latent_projection
            if bank == "state"
            else self.model_latent_projection
        )
        keys = (
            self.state_source_free_parent_keys
            if bank == "state"
            else self.model_source_free_parent_keys
        )
        biases = (
            self.state_source_free_biases
            if bank == "state"
            else self.model_source_free_biases
        )
        parameter_index = self._free_parameter_index(receiver_t)
        projected_content = projection(supported_latents)
        free_raw_scores = (
            projected_content[:-1] + keys[parameter_index]
        ) @ query + biases[parameter_index]
        complete_anchor_score = projected_content[-1] @ query
        return free_raw_scores - complete_anchor_score

    def _parent_specific_source_log_probs(
        self,
        *,
        prefix: CausalPrefix,
        earlier_latents: Tensor,
        bank: SourceBank,
    ) -> NormalizedSourceFactor:
        (
            receiver_t,
            query,
            supported_latents,
            parents,
            context_sha256,
        ) = self._checked_parent_specific_inputs(
            prefix=prefix,
            earlier_latents=earlier_latents,
            bank=bank,
        )
        free_logits = self._parent_specific_free_logits(
            receiver_t=receiver_t,
            query=query,
            supported_latents=supported_latents,
            parents=parents,
            bank=bank,
        )
        logits = _gauge_anchored_logits(
            free_logits=free_logits,
            receiver_t=receiver_t,
            parents=parents,
        )
        return self._normalized(
            logits=logits,
            prior_variant="parent_specific_pooled_prefix",
            bank=bank,
            receiver_t=receiver_t,
            context_sha256=context_sha256,
        )

    def state_source_log_probs(
        self, *, prefix: CausalPrefix, earlier_latents: Tensor
    ) -> NormalizedSourceFactor:
        return self._parent_specific_source_log_probs(
            prefix=prefix,
            earlier_latents=earlier_latents,
            bank="state",
        )

    def model_source_log_probs(
        self, *, prefix: CausalPrefix, earlier_latents: Tensor
    ) -> NormalizedSourceFactor:
        return self._parent_specific_source_log_probs(
            prefix=prefix,
            earlier_latents=earlier_latents,
            bank="model",
        )


__all__ = [
    "FixedSourcePrior",
    "FixedSourceFactorContext",
    "MaskCaseKey",
    "NormalizedSourceFactor",
    "ParentSpecificPooledPrefixSourcePrior",
    "PooledHistoryConditionedSourcePrior",
    "PrefixConditionedSourceFactorContext",
    "SourceFactorContext",
]
