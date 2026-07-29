"""H7-only exact fixed-source assembly specifications and A5 factory."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from vfe4.types.h6 import (
    ArmConfig,
    ArmId,
    H6LanguageStructure,
    canonical_json_bytes,
)

from .arms import BuiltArm


H7SourceFixtureId = Literal["h1-v1", "h7-v1"]
SourceProbabilityRows = tuple[tuple[float, ...], tuple[float, ...]]
SourceLogitRows = tuple[tuple[float, ...], tuple[float, ...]]

_H1_SOURCE_ROWS = ((0,), (0, 1))
_H7_V1_SOURCE_ROWS = ((0,), (1,))
_H1_STATE_SOURCE_PROBABILITIES = ((1.0,), (0.55, 0.45))
_H1_MODEL_SOURCE_PROBABILITIES = ((1.0,), (0.35, 0.65))
_H7_V1_SOURCE_PROBABILITIES = ((1.0,), (1.0,))
_SOURCE_SPECIFICATION_HASH_DOMAIN = "vfe4.h7.fixed-source-assembly-specification.v1"


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _source_rows(
    structure: H6LanguageStructure,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(structure) is not H6LanguageStructure:
        raise ValueError("structure must be an exact H6LanguageStructure")
    structure.__post_init__()
    if structure.receiver_labels != (1, 2) or len(structure.dag.rows) != 2:
        raise ValueError("H7 source structure must declare exactly receivers (1, 2)")
    return tuple(row.parents for row in structure.dag.rows)  # type: ignore[return-value]


def _require_fixture_source_rows(
    fixture_id: H7SourceFixtureId,
    source_rows: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    if fixture_id == "h1-v1":
        if source_rows != _H1_SOURCE_ROWS:
            raise ValueError("H1 source rows must be ((0,), (0, 1))")
    elif fixture_id == "h7-v1":
        if source_rows != _H7_V1_SOURCE_ROWS:
            raise ValueError("H7-v1 source rows must be ((0,), (1,))")
    else:
        raise ValueError("unsupported H7 source fixture identity")


def _checked_probabilities(
    rows: object,
    *,
    structure: H6LanguageStructure,
    name: str,
) -> SourceProbabilityRows:
    if type(rows) is not tuple or len(rows) != 2:
        raise ValueError(f"{name} must contain exactly two source rows")
    checked: list[tuple[float, ...]] = []
    for probabilities, dag_row in zip(rows, structure.dag.rows, strict=True):
        if (
            type(probabilities) is not tuple
            or len(probabilities) != len(dag_row.parents)
            or any(
                type(probability) is not float
                or not math.isfinite(probability)
                or probability <= 0.0
                for probability in probabilities
            )
            or math.fsum(probabilities) != 1.0
        ):
            raise ValueError(
                f"{name} rows must be exact positive normalized float tuples "
                "over the declared parents"
            )
        checked.append(probabilities)
    return (checked[0], checked[1])


def _gauge_anchored_logits(
    probabilities: SourceProbabilityRows,
    *,
    structure: H6LanguageStructure,
) -> SourceLogitRows:
    rows: list[tuple[float, ...]] = []
    for probability_row, dag_row in zip(probabilities, structure.dag.rows, strict=True):
        logits = [0.0] * dag_row.receiver_t
        anchor_log_probability = math.log(probability_row[-1])
        for parent, probability in zip(
            dag_row.parents[:-1], probability_row[:-1], strict=True
        ):
            logits[parent] = math.log(probability) - anchor_log_probability
        rows.append(tuple(logits))
    return (rows[0], rows[1])


def _specification_payload(
    *,
    fixture_id: H7SourceFixtureId,
    structure: H6LanguageStructure,
    state_source_probabilities: SourceProbabilityRows,
    model_source_probabilities: SourceProbabilityRows,
    state_logits: SourceLogitRows,
    model_logits: SourceLogitRows,
) -> dict[str, object]:
    return {
        "schema_version": "h7-fixed-source-assembly-specification-v1",
        "fixture_id": fixture_id,
        "structure_sha256": structure.structure_sha256,
        "source_rows": tuple(
            (row.receiver_t, row.parents) for row in structure.dag.rows
        ),
        "state_source_probabilities": state_source_probabilities,
        "model_source_probabilities": model_source_probabilities,
        "state_logits": state_logits,
        "model_logits": model_logits,
    }


@dataclass(frozen=True, slots=True)
class H7FixedSourceAssemblySpec:
    """Immutable H1/H7-v1 source law used only by the H7 assembly path."""

    fixture_id: H7SourceFixtureId
    structure: H6LanguageStructure
    structure_sha256: str
    state_source_probabilities: SourceProbabilityRows
    model_source_probabilities: SourceProbabilityRows
    state_logits: SourceLogitRows
    model_logits: SourceLogitRows
    source_specification_sha256: str

    def __post_init__(self) -> None:
        source_rows = _source_rows(self.structure)
        _require_fixture_source_rows(self.fixture_id, source_rows)
        if self.structure_sha256 != self.structure.structure_sha256:
            raise ValueError("source specification structure hash is stale")
        state_probabilities = _checked_probabilities(
            self.state_source_probabilities,
            structure=self.structure,
            name="state_source_probabilities",
        )
        model_probabilities = _checked_probabilities(
            self.model_source_probabilities,
            structure=self.structure,
            name="model_source_probabilities",
        )
        if self.fixture_id == "h1-v1":
            if (
                state_probabilities != _H1_STATE_SOURCE_PROBABILITIES
                or model_probabilities != _H1_MODEL_SOURCE_PROBABILITIES
            ):
                raise ValueError("H1 fixed source probabilities changed")
        else:
            if (
                state_probabilities != _H7_V1_SOURCE_PROBABILITIES
                or model_probabilities != _H7_V1_SOURCE_PROBABILITIES
            ):
                raise ValueError("H7-v1 fixed source probabilities changed")
        expected_state_logits = _gauge_anchored_logits(
            state_probabilities,
            structure=self.structure,
        )
        expected_model_logits = _gauge_anchored_logits(
            model_probabilities,
            structure=self.structure,
        )
        if self.state_logits != expected_state_logits:
            raise ValueError("state logits do not match the fixed source probabilities")
        if self.model_logits != expected_model_logits:
            raise ValueError("model logits do not match the fixed source probabilities")
        expected_hash = _owned_hash(
            _SOURCE_SPECIFICATION_HASH_DOMAIN,
            _specification_payload(
                fixture_id=self.fixture_id,
                structure=self.structure,
                state_source_probabilities=state_probabilities,
                model_source_probabilities=model_probabilities,
                state_logits=expected_state_logits,
                model_logits=expected_model_logits,
            ),
        )
        if self.source_specification_sha256 != expected_hash:
            raise ValueError(
                "source specification hash does not match structure and probabilities"
            )

    @classmethod
    def from_probabilities(
        cls,
        *,
        fixture_id: H7SourceFixtureId,
        structure: H6LanguageStructure,
        state_source_probabilities: SourceProbabilityRows,
        model_source_probabilities: SourceProbabilityRows,
    ) -> H7FixedSourceAssemblySpec:
        """Canonicalize exact probabilities into fixed-zero-gauge source logits."""

        source_rows = _source_rows(structure)
        _require_fixture_source_rows(fixture_id, source_rows)
        state_probabilities = _checked_probabilities(
            state_source_probabilities,
            structure=structure,
            name="state_source_probabilities",
        )
        model_probabilities = _checked_probabilities(
            model_source_probabilities,
            structure=structure,
            name="model_source_probabilities",
        )
        state_logits = _gauge_anchored_logits(
            state_probabilities,
            structure=structure,
        )
        model_logits = _gauge_anchored_logits(
            model_probabilities,
            structure=structure,
        )
        payload = _specification_payload(
            fixture_id=fixture_id,
            structure=structure,
            state_source_probabilities=state_probabilities,
            model_source_probabilities=model_probabilities,
            state_logits=state_logits,
            model_logits=model_logits,
        )
        return cls(
            fixture_id=fixture_id,
            structure=structure,
            structure_sha256=structure.structure_sha256,
            state_source_probabilities=state_probabilities,
            model_source_probabilities=model_probabilities,
            state_logits=state_logits,
            model_logits=model_logits,
            source_specification_sha256=_owned_hash(
                _SOURCE_SPECIFICATION_HASH_DOMAIN,
                payload,
            ),
        )

    @classmethod
    def from_h1(
        cls,
        structure: H6LanguageStructure,
    ) -> H7FixedSourceAssemblySpec:
        """Bind the frozen H1 dense rows and nonuniform generative source law."""

        return cls.from_probabilities(
            fixture_id="h1-v1",
            structure=structure,
            state_source_probabilities=_H1_STATE_SOURCE_PROBABILITIES,
            model_source_probabilities=_H1_MODEL_SOURCE_PROBABILITIES,
        )

    @classmethod
    def from_h7_v1(
        cls,
        structure: H6LanguageStructure,
    ) -> H7FixedSourceAssemblySpec:
        """Bind the frozen H7-v1 singleton-predecessor source law."""

        return cls.from_probabilities(
            fixture_id="h7-v1",
            structure=structure,
            state_source_probabilities=_H7_V1_SOURCE_PROBABILITIES,
            model_source_probabilities=_H7_V1_SOURCE_PROBABILITIES,
        )


def _tensor_rows(rows: SourceLogitRows) -> tuple[Tensor, Tensor]:
    return tuple(torch.tensor(row, dtype=torch.float64) for row in rows)  # type: ignore[return-value]


def _validate_h7_a5_config(config: ArmConfig) -> None:
    if type(config) is not ArmConfig:
        raise ValueError("config must be an exact ArmConfig")
    config.__post_init__()
    if config.horizon != 2:
        raise ValueError("H7 assembly horizon must be exactly 2")
    if (
        config.arm is not ArmId.A5
        or not config.latent_enabled
        or not config.state_channel_enabled
        or not config.model_channel_enabled
        or config.source_mode != "categorical"
        or config.map_mode != "shared_vertex_coboundary"
        or config.prior_variant != "fixed"
        or config.mixture_mode != "exact"
        or config.objective_kind != "complete_elbo"
    ):
        raise ValueError(
            "H7 assembly requires an exact latent A5 fixed/exact/complete config"
        )


def build_h7_fixed_a5_arm(
    config: ArmConfig,
    source_spec: H7FixedSourceAssemblySpec,
) -> BuiltArm:
    """Issue one exact fixed/exact/complete A5 through the protected factory."""

    if type(source_spec) is not H7FixedSourceAssemblySpec:
        raise ValueError("source_spec must be an exact H7FixedSourceAssemblySpec")
    source_spec.__post_init__()
    _validate_h7_a5_config(config)
    from .arms import _construct_with_fixed_source_prior

    return _construct_with_fixed_source_prior(
        config,
        structure=source_spec.structure,
        source_specification_sha256=(source_spec.source_specification_sha256),
        state_logits=_tensor_rows(source_spec.state_logits),
        model_logits=_tensor_rows(source_spec.model_logits),
    )


__all__ = [
    "H7FixedSourceAssemblySpec",
    "H7SourceFixtureId",
    "SourceLogitRows",
    "SourceProbabilityRows",
    "build_h7_fixed_a5_arm",
]
