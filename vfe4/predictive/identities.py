"""Canonical semantic and byte identities for H6 prior prediction."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import nn

from vfe4.types.h6 import EstimatorSpec, VocabularyIdentity, canonical_json_bytes


_LOWER_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def vocabulary_identity_sha256(vocabulary: VocabularyIdentity) -> str:
    """Hash the complete semantic vocabulary identity in one canonical domain."""

    if type(vocabulary) is not VocabularyIdentity:
        raise ValueError("vocabulary must be an exact VocabularyIdentity")
    vocabulary.__post_init__()
    return _owned_hash(
        "vfe4.h6.vocabulary-identity.v1",
        {
            "vocabulary_id": vocabulary.vocabulary_id,
            "size": vocabulary.size,
            "tokenizer_spec_sha256": vocabulary.tokenizer_spec_sha256,
        },
    )


def _tensor_little_endian_bytes(value: torch.Tensor) -> bytes:
    if value.layout is not torch.strided or value.is_quantized:
        raise ValueError("model-state tensors must be dense, unquantized tensors")
    cpu = value.detach().to(device="cpu").contiguous()
    if (cpu.is_floating_point() or cpu.is_complex()) and not bool(
        torch.isfinite(cpu).all()
    ):
        raise ValueError("model-state tensors must be finite")
    raw = bytes(cpu.view(torch.uint8).reshape(-1).tolist())
    if sys.byteorder == "little" or cpu.element_size() == 1:
        return raw
    width = cpu.element_size()
    return b"".join(
        raw[offset : offset + width][::-1]
        for offset in range(0, len(raw), width)
    )


def canonical_model_state_sha256(model: nn.Module) -> str:
    """Hash named parameters and buffers with explicit dtype/shape/little-endian bytes."""

    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    state: Mapping[str, torch.Tensor] = model.state_dict()
    names = tuple(sorted(state))
    preimage = bytearray(b"VFE4-H6-MODEL-STATE-V1\x00")
    preimage.extend(len(names).to_bytes(4, "little"))
    for name in names:
        if type(name) is not str or not name:
            raise ValueError("model-state names must be nonempty strings")
        tensor = state[name]
        if type(tensor) is not torch.Tensor:
            raise ValueError("model state must contain only exact tensors")
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).removeprefix("torch.").encode("ascii")
        raw = _tensor_little_endian_bytes(tensor)
        preimage.extend(len(name_bytes).to_bytes(4, "little"))
        preimage.extend(name_bytes)
        preimage.extend(len(dtype_bytes).to_bytes(2, "little"))
        preimage.extend(dtype_bytes)
        preimage.extend(tensor.ndim.to_bytes(2, "little"))
        for size in tensor.shape:
            preimage.extend(int(size).to_bytes(8, "little"))
        preimage.extend(len(raw).to_bytes(8, "little"))
        preimage.extend(raw)
    return hashlib.sha256(bytes(preimage)).hexdigest()


def _estimator_semantic_payload(spec: EstimatorSpec) -> dict[str, object]:
    return {
        "schema_version": spec.schema_version,
        "kind": spec.kind,
        "particle_count": spec.particle_count,
        "resampling": spec.resampling,
        "dtype": spec.dtype,
        "device": spec.device,
    }


def _estimator_artifact_payload(spec: EstimatorSpec) -> dict[str, object]:
    return {
        **_estimator_semantic_payload(spec),
        "estimator_semantic_sha256": spec.estimator_sha256,
    }


@dataclass(frozen=True, init=False)
class EstimatorIdentity:
    """Keep estimator semantics distinct from the exact artifact byte identity."""

    semantic_sha256: str
    artifact_bytes_sha256: str
    identity_sha256: str
    _artifact_bytes: bytes = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha256(self.semantic_sha256, "semantic_sha256")
        _require_sha256(self.artifact_bytes_sha256, "artifact_bytes_sha256")
        _require_sha256(self.identity_sha256, "identity_sha256")
        if type(self._artifact_bytes) is not bytes:
            raise ValueError("estimator artifact bytes must be immutable bytes")
        if hashlib.sha256(self._artifact_bytes).hexdigest() != self.artifact_bytes_sha256:
            raise ValueError("estimator artifact byte identity does not match its bytes")
        expected = _owned_hash(
            "vfe4.h6.estimator-identity.v1",
            {
                "semantic_sha256": self.semantic_sha256,
                "artifact_bytes_sha256": self.artifact_bytes_sha256,
            },
        )
        if self.identity_sha256 != expected:
            raise ValueError("estimator identity does not match semantic and byte fields")

    @classmethod
    def from_spec(
        cls,
        spec: EstimatorSpec,
        *,
        artifact_bytes: bytes | None = None,
    ) -> "EstimatorIdentity":
        if type(spec) is not EstimatorSpec:
            raise ValueError("spec must be an exact EstimatorSpec")
        spec.__post_init__()
        semantic_payload = _estimator_semantic_payload(spec)
        semantic_sha256 = _owned_hash(
            "vfe4.h6.estimator-spec.v1", semantic_payload
        )
        if semantic_sha256 != spec.estimator_sha256:
            raise ValueError("EstimatorSpec semantic identity is stale")
        expected_bytes = canonical_json_bytes(_estimator_artifact_payload(spec))
        if artifact_bytes is None:
            checked_bytes = expected_bytes
        else:
            if type(artifact_bytes) is not bytes or artifact_bytes != expected_bytes:
                raise ValueError(
                    "estimator artifact bytes must be the exact canonical spec artifact"
                )
            checked_bytes = artifact_bytes
        artifact_sha256 = hashlib.sha256(checked_bytes).hexdigest()
        identity_sha256 = _owned_hash(
            "vfe4.h6.estimator-identity.v1",
            {
                "semantic_sha256": semantic_sha256,
                "artifact_bytes_sha256": artifact_sha256,
            },
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "semantic_sha256", semantic_sha256)
        object.__setattr__(instance, "artifact_bytes_sha256", artifact_sha256)
        object.__setattr__(instance, "identity_sha256", identity_sha256)
        object.__setattr__(instance, "_artifact_bytes", checked_bytes)
        instance.__post_init__()
        return instance

    @property
    def artifact_bytes(self) -> bytes:
        self.__post_init__()
        return bytes(self._artifact_bytes)


__all__ = [
    "EstimatorIdentity",
    "canonical_model_state_sha256",
    "vocabulary_identity_sha256",
]
