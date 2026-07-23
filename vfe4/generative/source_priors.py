"""Normalized causal source priors for the H6 language model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
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


PriorVariant = Literal["fixed", "prefix_conditioned"]
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
        if self.prior_variant not in ("fixed", "prefix_conditioned"):
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


@dataclass(frozen=True)
class NormalizedSourceFactor:
    """One normalized source row with immutable value and support identities."""

    mask_case_key: MaskCaseKey
    support_mask: tuple[bool, ...]
    log_probs: FrozenTensorSnapshot
    factor_identity_sha256: str

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
            {
                "mask_case_sha256": self.mask_case_key.canonical_sha256,
                "support_mask": self.support_mask,
                "log_probs": _snapshot_payload(self.log_probs),
            },
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


def _checked_rows(
    rows: tuple[Tensor, ...], structure: H6LanguageStructure, name: str
) -> nn.ParameterList:
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
        parameters.append(nn.Parameter(row.detach().clone()))
    return nn.ParameterList(parameters)


class FixedSourcePrior(_SourcePriorBase):
    """Trainable prefix-independent normalized priors for both source banks."""

    def __init__(
        self,
        *,
        structure: H6LanguageStructure,
        vocabulary: VocabularyIdentity,
        fixture_sha256: str,
        predictor_config_sha256: str,
        model_family_sha256: str,
        state_logits: tuple[Tensor, ...],
        model_logits: tuple[Tensor, ...],
    ) -> None:
        super().__init__(
            structure=structure,
            vocabulary=vocabulary,
            fixture_sha256=fixture_sha256,
            predictor_config_sha256=predictor_config_sha256,
            model_family_sha256=model_family_sha256,
        )
        self.state_logits = _checked_rows(state_logits, structure, "state_logits")
        self.model_logits = _checked_rows(model_logits, structure, "model_logits")

    def _context_sha256(self, receiver_t: int) -> str:
        return _owned_hash(
            "vfe4.h6.fixed-source-context.v1", {"receiver_t": receiver_t}
        )

    def state_source_log_probs(self, *, receiver_t: int) -> NormalizedSourceFactor:
        index = self._row_index(receiver_t)
        return self._normalized(
            logits=self.state_logits[index],
            prior_variant="fixed",
            bank="state",
            receiver_t=receiver_t,
            context_sha256=self._context_sha256(receiver_t),
        )

    def model_source_log_probs(self, *, receiver_t: int) -> NormalizedSourceFactor:
        index = self._row_index(receiver_t)
        return self._normalized(
            logits=self.model_logits[index],
            prior_variant="fixed",
            bank="model",
            receiver_t=receiver_t,
            context_sha256=self._context_sha256(receiver_t),
        )


class PrefixConditionedSourcePrior(_SourcePriorBase):
    """Normalized priors conditioned only on prior tokens and generated history."""

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
        self.state_parent_keys = nn.ParameterList(
            [
                nn.Parameter(torch.zeros((row.receiver_t, context_dim), dtype=torch.float64))
                for row in structure.dag.rows
            ]
        )
        self.model_parent_keys = nn.ParameterList(
            [
                nn.Parameter(torch.zeros((row.receiver_t, context_dim), dtype=torch.float64))
                for row in structure.dag.rows
            ]
        )
        self.state_biases = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(row.receiver_t, dtype=torch.float64))
                for row in structure.dag.rows
            ]
        )
        self.model_biases = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(row.receiver_t, dtype=torch.float64))
                for row in structure.dag.rows
            ]
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
            "vfe4.h6.prefix-source-context.v1",
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
        logits = self.state_parent_keys[index] @ context + self.state_biases[index]
        return self._normalized(
            logits=logits,
            prior_variant="prefix_conditioned",
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
        logits = self.model_parent_keys[index] @ context + self.model_biases[index]
        return self._normalized(
            logits=logits,
            prior_variant="prefix_conditioned",
            bank="model",
            receiver_t=receiver_t,
            context_sha256=context_sha256,
        )


__all__ = [
    "FixedSourcePrior",
    "MaskCaseKey",
    "NormalizedSourceFactor",
    "PrefixConditionedSourcePrior",
]
