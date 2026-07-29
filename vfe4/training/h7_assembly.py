"""H7-only exact fixed-source assembly specifications and A5 factory."""

from __future__ import annotations

import hashlib
import math
import weakref
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from vfe4.generative.source_priors import FixedSourcePrior
from vfe4.predictive.identities import canonical_model_state_sha256
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
H7AssemblySourceRow = tuple[
    str,
    int,
    tuple[int, ...],
    tuple[float, ...],
    str,
]

_H1_SOURCE_ROWS = ((0,), (0, 1))
_H7_V1_SOURCE_ROWS = ((0,), (1,))
_H1_STATE_SOURCE_PROBABILITIES = ((1.0,), (0.55, 0.45))
_H1_MODEL_SOURCE_PROBABILITIES = ((1.0,), (0.35, 0.65))
_H7_V1_SOURCE_PROBABILITIES = ((1.0,), (1.0,))
_SOURCE_SPECIFICATION_HASH_DOMAIN = "vfe4.h7.fixed-source-assembly-specification.v1"
_SOURCE_LAW_HASH_DOMAIN = "vfe4.h7.fixed-source-assembly-live-law.v1"
_ASSEMBLY_RECEIPT_HASH_DOMAIN = "vfe4.h7.fixed-source-assembly-receipt.v1"


def _owned_hash(domain: str, payload: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + canonical_json_bytes(payload)
    ).hexdigest()


def _source_rows(
    structure: H6LanguageStructure,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(structure) is not H6LanguageStructure:
        raise ValueError("structure must be an exact H6LanguageStructure")
    structure.base.__post_init__()
    for row in structure.dag.rows:
        row.__post_init__()
    structure.dag.__post_init__()
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


@dataclass(frozen=True, slots=True, init=False)
class H7FixedSourceAssemblyReceipt:
    """Registry-backed evidence for one exact H1/H7 fixed-source arm."""

    schema_version: Literal["h7-fixed-source-assembly-receipt-v1"]
    fixture_id: H7SourceFixtureId
    source_specification_sha256: str
    structure_sha256: str
    endpoint_config_sha256: str
    model_family_sha256: str
    model_state_sha256: str
    proposal_identity_sha256: str
    predictor_config_sha256: str
    parameter_role_sha256s: tuple[str, ...]
    source_rows: tuple[
        H7AssemblySourceRow,
        H7AssemblySourceRow,
        H7AssemblySourceRow,
        H7AssemblySourceRow,
    ]
    source_law_sha256: str
    assembly_sha256: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fixture_id": self.fixture_id,
            "source_specification_sha256": (
                self.source_specification_sha256
            ),
            "structure_sha256": self.structure_sha256,
            "endpoint_config_sha256": self.endpoint_config_sha256,
            "model_family_sha256": self.model_family_sha256,
            "model_state_sha256": self.model_state_sha256,
            "proposal_identity_sha256": self.proposal_identity_sha256,
            "predictor_config_sha256": self.predictor_config_sha256,
            "parameter_role_sha256s": self.parameter_role_sha256s,
            "source_rows": self.source_rows,
            "source_law_sha256": self.source_law_sha256,
        }

    def __post_init__(self) -> None:
        if (
            self.schema_version
            != "h7-fixed-source-assembly-receipt-v1"
            or self.fixture_id not in ("h1-v1", "h7-v1")
        ):
            raise ValueError("H7 fixed-source receipt identity changed")
        for name in (
            "source_specification_sha256",
            "structure_sha256",
            "endpoint_config_sha256",
            "model_family_sha256",
            "model_state_sha256",
            "proposal_identity_sha256",
            "predictor_config_sha256",
            "source_law_sha256",
            "assembly_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            type(self.parameter_role_sha256s) is not tuple
            or not self.parameter_role_sha256s
            or len(set(self.parameter_role_sha256s))
            != len(self.parameter_role_sha256s)
        ):
            raise ValueError("H7 assembly parameter-role inventory is invalid")
        for value in self.parameter_role_sha256s:
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(
                    "H7 assembly parameter-role hashes must be SHA-256 digests"
                )
        expected_row_order = (
            ("model_source", 1),
            ("state_source", 1),
            ("model_source", 2),
            ("state_source", 2),
        )
        if (
            type(self.source_rows) is not tuple
            or len(self.source_rows) != 4
            or tuple(row[:2] for row in self.source_rows)
            != expected_row_order
        ):
            raise ValueError("H7 assembly source-row inventory is invalid")
        for partition, receiver_t, support, probabilities, factor_sha256 in (
            self.source_rows
        ):
            if (
                partition not in ("model_source", "state_source")
                or receiver_t not in (1, 2)
                or type(support) is not tuple
                or not support
                or type(probabilities) is not tuple
                or len(probabilities) != len(support)
                or any(
                    type(value) is not float
                    or not math.isfinite(value)
                    or value <= 0.0
                    for value in probabilities
                )
                or math.fsum(probabilities) != 1.0
                or type(factor_sha256) is not str
                or len(factor_sha256) != 64
            ):
                raise ValueError("H7 assembly source row is malformed")
        if self.source_law_sha256 != _owned_hash(
            _SOURCE_LAW_HASH_DOMAIN,
            self.source_rows,
        ):
            raise ValueError("H7 assembly live source-law hash changed")
        if self.assembly_sha256 != _owned_hash(
            _ASSEMBLY_RECEIPT_HASH_DOMAIN,
            self.canonical_payload(),
        ):
            raise ValueError("H7 fixed-source assembly receipt hash changed")


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


def _live_source_rows(arm: BuiltArm) -> tuple[
    H7AssemblySourceRow,
    H7AssemblySourceRow,
    H7AssemblySourceRow,
    H7AssemblySourceRow,
]:
    from .arms import LatentLanguageArmModel

    if type(arm) is not BuiltArm or type(arm.model) is not LatentLanguageArmModel:
        raise ValueError("H7-specific assembly requires an exact latent BuiltArm")
    prior = arm.model.source_prior
    if type(prior) is not FixedSourcePrior:
        raise ValueError("H7-specific assembly requires an exact fixed source prior")
    _source_rows(prior.structure)
    result: list[H7AssemblySourceRow] = []
    for receiver_t in (1, 2):
        for partition in ("model_source", "state_source"):
            factor = (
                prior.model_source_log_probs(receiver_t=receiver_t)
                if partition == "model_source"
                else prior.state_source_log_probs(receiver_t=receiver_t)
            )
            factor.__post_init__()
            support = tuple(
                index
                for index, enabled in enumerate(factor.support_mask)
                if enabled
            )
            values = factor.log_probs.value()
            probabilities = tuple(
                float(torch.exp(values[index]).item()) for index in support
            )
            result.append(
                (
                    partition,
                    receiver_t,
                    support,
                    probabilities,
                    factor.factor_identity_sha256,
                )
            )
    return tuple(result)  # type: ignore[return-value]


def _require_rows_match_spec(
    rows: tuple[
        H7AssemblySourceRow,
        H7AssemblySourceRow,
        H7AssemblySourceRow,
        H7AssemblySourceRow,
    ],
    spec: H7FixedSourceAssemblySpec,
) -> None:
    expected_probabilities = {
        ("model_source", 1): spec.model_source_probabilities[0],
        ("state_source", 1): spec.state_source_probabilities[0],
        ("model_source", 2): spec.model_source_probabilities[1],
        ("state_source", 2): spec.state_source_probabilities[1],
    }
    expected_supports = {
        (partition, receiver_t): spec.structure.dag.rows[
            receiver_t - 1
        ].parents
        for receiver_t in (1, 2)
        for partition in ("model_source", "state_source")
    }
    for partition, receiver_t, support, probabilities, _ in rows:
        key = (partition, receiver_t)
        expected = expected_probabilities[key]
        if support != expected_supports[key] or len(probabilities) != len(expected):
            raise ValueError(
                "live fixed-source support differs from its H7 specification"
            )
        allowance = 4.0 * math.ulp(1.0) * max(1, len(expected))
        if any(
            not math.isclose(
                observed,
                frozen,
                rel_tol=0.0,
                abs_tol=allowance,
            )
            for observed, frozen in zip(
                probabilities,
                expected,
                strict=True,
            )
        ):
            raise ValueError(
                "live fixed-source probabilities differ from their H7 "
                "specification"
            )


def _new_assembly_receipt(
    arm: BuiltArm,
    *,
    fixture_id: H7SourceFixtureId,
    source_specification_sha256: str,
    structure_sha256: str,
) -> H7FixedSourceAssemblyReceipt:
    rows = _live_source_rows(arm)
    source_law_sha256 = _owned_hash(_SOURCE_LAW_HASH_DOMAIN, rows)
    parameter_role_sha256s = tuple(
        record.record_sha256 for record in arm.parameter_roles
    )
    values: dict[str, object] = {
        "schema_version": "h7-fixed-source-assembly-receipt-v1",
        "fixture_id": fixture_id,
        "source_specification_sha256": source_specification_sha256,
        "structure_sha256": structure_sha256,
        "endpoint_config_sha256": arm.config.config_sha256,
        "model_family_sha256": arm.model_family_sha256,
        "model_state_sha256": canonical_model_state_sha256(arm.model),
        "proposal_identity_sha256": arm.proposal.proposal_identity_sha256,
        "predictor_config_sha256": arm.predictor.predictor_config_sha256,
        "parameter_role_sha256s": parameter_role_sha256s,
        "source_rows": rows,
        "source_law_sha256": source_law_sha256,
    }
    receipt = object.__new__(H7FixedSourceAssemblyReceipt)
    for name, value in values.items():
        object.__setattr__(receipt, name, value)
    object.__setattr__(
        receipt,
        "assembly_sha256",
        _owned_hash(_ASSEMBLY_RECEIPT_HASH_DOMAIN, receipt.canonical_payload()),
    )
    receipt.__post_init__()
    return receipt


def _assembly_relationship_snapshot(
    arm: BuiltArm,
    receipt: H7FixedSourceAssemblyReceipt,
) -> tuple[object, ...]:
    from .arms import LatentLanguageArmModel

    receipt.__post_init__()
    if type(arm) is not BuiltArm or type(arm.model) is not LatentLanguageArmModel:
        raise ValueError("H7 assembly arm relationship is invalid")
    prior = arm.model.source_prior
    if (
        type(prior) is not FixedSourcePrior
        or arm.proposal.model is not arm.model
        or arm.predictor.proposal is not arm.proposal
        or arm.proposal.model_family_sha256 != arm.model_family_sha256
        or arm.predictor.model_family_sha256 != arm.model_family_sha256
        or arm.proposal.model_state_sha256 != receipt.model_state_sha256
        or arm.predictor.model_state_sha256 != receipt.model_state_sha256
        or prior.fixture_sha256 != receipt.source_specification_sha256
        or prior.structure.structure_sha256 != receipt.structure_sha256
        or prior.predictor_config_sha256 != arm.config.config_sha256
        or prior.model_family_sha256 != arm.model_family_sha256
    ):
        raise ValueError("H7 assembly arm relationship changed")
    arm.proposal.assert_current_state()
    for record in arm.parameter_roles:
        record.__post_init__()
    for binding in arm.optimizer_bindings:
        binding.__post_init__()
    for term in arm.flop_terms:
        term.__post_init__()
    return (
        id(arm.config),
        id(arm.model),
        id(arm.recognition_store),
        id(arm.proposal),
        id(arm.predictor),
        id(prior),
        id(prior.structure),
        id(arm.proposal.model),
        id(arm.predictor.proposal),
        canonical_model_state_sha256(arm.model),
        receipt.assembly_sha256,
        tuple(record.record_sha256 for record in arm.parameter_roles),
        tuple(binding.binding_sha256 for binding in arm.optimizer_bindings),
        tuple(term.term_sha256 for term in arm.flop_terms),
        arm.elbo_inventory_sha256,
        tuple(arm.elbo_factor_inventory),
    )


def _build_h7_fixed_source_assembly_api():
    registry: dict[
        int,
        tuple[
            weakref.ReferenceType[BuiltArm],
            H7FixedSourceAssemblyReceipt,
            bytes,
            tuple[object, ...],
        ],
    ] = {}

    def build(
        config: ArmConfig,
        source_spec: H7FixedSourceAssemblySpec,
    ) -> BuiltArm:
        if type(source_spec) is not H7FixedSourceAssemblySpec:
            raise ValueError(
                "source_spec must be an exact H7FixedSourceAssemblySpec"
            )
        _source_rows(source_spec.structure)
        source_spec.__post_init__()
        _validate_h7_a5_config(config)
        from .arms import _construct_with_fixed_source_prior

        arm = _construct_with_fixed_source_prior(
            config,
            structure=source_spec.structure,
            source_specification_sha256=(
                source_spec.source_specification_sha256
            ),
            state_logits=_tensor_rows(source_spec.state_logits),
            model_logits=_tensor_rows(source_spec.model_logits),
        )
        rows = _live_source_rows(arm)
        _require_rows_match_spec(rows, source_spec)
        receipt = _new_assembly_receipt(
            arm,
            fixture_id=source_spec.fixture_id,
            source_specification_sha256=(
                source_spec.source_specification_sha256
            ),
            structure_sha256=source_spec.structure_sha256,
        )
        snapshot = _assembly_relationship_snapshot(arm, receipt)
        identity = id(arm)

        def remove(reference: weakref.ReferenceType[BuiltArm]) -> None:
            current = registry.get(identity)
            if current is not None and current[0] is reference:
                registry.pop(identity, None)

        reference = weakref.ref(arm, remove)
        if identity in registry and registry[identity][0]() is not None:
            raise RuntimeError("H7-specific arm identity was already issued")
        registry[identity] = (
            reference,
            receipt,
            canonical_json_bytes(receipt.canonical_payload()),
            snapshot,
        )
        return arm

    def require(arm: object) -> H7FixedSourceAssemblyReceipt:
        if type(arm) is not BuiltArm:
            raise ValueError("H7-specific assembly requires an exact BuiltArm")
        current = registry.get(id(arm))
        if current is None or current[0]() is not arm:
            raise ValueError(
                "arm lacks H7-specific fixed-source assembly issuance"
            )
        receipt = current[1]
        try:
            receipt.__post_init__()
            if (
                canonical_json_bytes(receipt.canonical_payload()) != current[2]
                or _assembly_relationship_snapshot(arm, receipt) != current[3]
            ):
                raise ValueError(
                    "H7-specific fixed-source assembly changed after issuance"
                )
            observed = _new_assembly_receipt(
                arm,
                fixture_id=receipt.fixture_id,
                source_specification_sha256=(
                    receipt.source_specification_sha256
                ),
                structure_sha256=receipt.structure_sha256,
            )
        except ValueError as exc:
            raise ValueError(
                "H7-specific source integrity changed after issuance"
            ) from exc
        if observed != receipt:
            raise ValueError("H7-specific live source law changed after issuance")
        return receipt

    return build, require


(
    build_h7_fixed_a5_arm,
    require_h7_fixed_source_assembly,
) = _build_h7_fixed_source_assembly_api()
del _build_h7_fixed_source_assembly_api


__all__ = [
    "H7AssemblySourceRow",
    "H7FixedSourceAssemblySpec",
    "H7FixedSourceAssemblyReceipt",
    "H7SourceFixtureId",
    "SourceLogitRows",
    "SourceProbabilityRows",
    "build_h7_fixed_a5_arm",
    "require_h7_fixed_source_assembly",
]
