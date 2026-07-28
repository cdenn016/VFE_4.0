"""Exclusive durable WikiText-103 held-out-test opening authority."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vfe4.artifacts.durability import (
    DurabilityBackend,
    DurableFileIdentity,
    canonical_json_bytes_generic,
)
from vfe4.data.windows import CausalWindowSet
from vfe4.types import (
    EndpointInventory,
    WT103CheckpointIdentity,
    validate_endpoint_inventory,
)
from vfe4.types.training import owned_sha256


_HEX = frozenset("0123456789abcdef")
_CANONICAL_RESERVATION_PREFIX = ".vfe4-test-opening-"


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_reservation_path(
    repository_root: Path,
    reservation_scope_sha256: str,
) -> Path:
    _require_sha256(
        reservation_scope_sha256,
        "reservation_scope_sha256",
    )
    return (
        repository_root
        / (
            f"{_CANONICAL_RESERVATION_PREFIX}"
            f"{reservation_scope_sha256}.reservation.json"
        )
    )


def _opening_scope_sha256(
    *,
    endpoint_inventory: EndpointInventory,
    run_group_manifest_sha256: str,
    data_identity_sha256: str,
    tokenizer_identity_sha256: str,
    test_window_manifest_sha256: str,
    test_schedule_sha256: str,
) -> str:
    return owned_sha256(
        "vfe4.wt103.test-opening-scope.v1",
        {
            "endpoint_inventory_sha256": (
                endpoint_inventory.endpoint_inventory_sha256
            ),
            "run_group_manifest_sha256": run_group_manifest_sha256,
            "data_identity_sha256": data_identity_sha256,
            "tokenizer_identity_sha256": tokenizer_identity_sha256,
            "test_window_manifest_sha256": test_window_manifest_sha256,
            "test_schedule_sha256": test_schedule_sha256,
        },
    )


def _opening_plan_payload(
    *,
    endpoint_inventory: EndpointInventory,
    terminal_checkpoints: tuple[WT103CheckpointIdentity, ...],
    run_group_complete: bool,
    run_group_manifest_sha256: str,
    analysis_sha256: str,
    figure_sha256: str,
    data_identity_sha256: str,
    tokenizer_identity_sha256: str,
    test_window_manifest_sha256: str,
    test_schedule_sha256: str,
) -> dict[str, object]:
    reservation_scope_sha256 = _opening_scope_sha256(
        endpoint_inventory=endpoint_inventory,
        run_group_manifest_sha256=run_group_manifest_sha256,
        data_identity_sha256=data_identity_sha256,
        tokenizer_identity_sha256=tokenizer_identity_sha256,
        test_window_manifest_sha256=test_window_manifest_sha256,
        test_schedule_sha256=test_schedule_sha256,
    )
    return {
        "schema_version": "wt103-test-opening-plan-v1",
        "endpoint_inventory_sha256": (
            endpoint_inventory.endpoint_inventory_sha256
        ),
        "terminal_checkpoint_keys": tuple(
            checkpoint.logical_key for checkpoint in terminal_checkpoints
        ),
        "terminal_checkpoint_identity_sha256s": tuple(
            checkpoint.checkpoint_identity_sha256
            for checkpoint in terminal_checkpoints
        ),
        "run_group_complete": run_group_complete,
        "run_group_manifest_sha256": run_group_manifest_sha256,
        "analysis_sha256": analysis_sha256,
        "figure_sha256": figure_sha256,
        "estimator_protocol_sha256": (
            endpoint_inventory.estimator_protocol_sha256
        ),
        "data_identity_sha256": data_identity_sha256,
        "tokenizer_identity_sha256": tokenizer_identity_sha256,
        "test_window_manifest_sha256": test_window_manifest_sha256,
        "test_schedule_sha256": test_schedule_sha256,
        "reservation_scope_sha256": reservation_scope_sha256,
    }


@dataclass(frozen=True, slots=True)
class TestOpeningPlan:
    repository_root: Path
    reservation_path: Path
    durability_backend: DurabilityBackend
    endpoint_inventory: EndpointInventory
    terminal_checkpoints: tuple[WT103CheckpointIdentity, ...]
    run_group_complete: bool
    run_group_manifest_sha256: str
    analysis_sha256: str
    figure_sha256: str
    data_identity_sha256: str
    tokenizer_identity_sha256: str
    test_window_manifest_sha256: str
    test_schedule_sha256: str
    reservation_scope_sha256: str
    opening_plan_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "run_group_manifest_sha256",
            "analysis_sha256",
            "figure_sha256",
            "data_identity_sha256",
            "tokenizer_identity_sha256",
            "test_window_manifest_sha256",
            "test_schedule_sha256",
            "reservation_scope_sha256",
            "opening_plan_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        expected = owned_sha256(
            "vfe4.wt103.test-opening-plan.v1",
            _opening_plan_payload(
                endpoint_inventory=self.endpoint_inventory,
                terminal_checkpoints=self.terminal_checkpoints,
                run_group_complete=self.run_group_complete,
                run_group_manifest_sha256=self.run_group_manifest_sha256,
                analysis_sha256=self.analysis_sha256,
                figure_sha256=self.figure_sha256,
                data_identity_sha256=self.data_identity_sha256,
                tokenizer_identity_sha256=self.tokenizer_identity_sha256,
                test_window_manifest_sha256=self.test_window_manifest_sha256,
                test_schedule_sha256=self.test_schedule_sha256,
            ),
        )
        expected_scope = _opening_scope_sha256(
            endpoint_inventory=self.endpoint_inventory,
            run_group_manifest_sha256=self.run_group_manifest_sha256,
            data_identity_sha256=self.data_identity_sha256,
            tokenizer_identity_sha256=self.tokenizer_identity_sha256,
            test_window_manifest_sha256=self.test_window_manifest_sha256,
            test_schedule_sha256=self.test_schedule_sha256,
        )
        if self.reservation_scope_sha256 != expected_scope:
            raise ValueError(
                "reservation_scope_sha256 does not match the run group"
            )
        if self.opening_plan_sha256 != expected:
            raise ValueError(
                "opening_plan_sha256 does not match the opening plan"
            )

    @classmethod
    def create(
        cls,
        *,
        repository_root: Path,
        durability_backend: DurabilityBackend,
        endpoint_inventory: EndpointInventory,
        terminal_checkpoints: tuple[WT103CheckpointIdentity, ...],
        run_group_complete: bool,
        run_group_manifest_sha256: str,
        analysis_sha256: str,
        figure_sha256: str,
        data_identity_sha256: str,
        tokenizer_identity_sha256: str,
        test_window_manifest_sha256: str,
        test_schedule_sha256: str,
    ) -> "TestOpeningPlan":
        payload = _opening_plan_payload(
            endpoint_inventory=endpoint_inventory,
            terminal_checkpoints=terminal_checkpoints,
            run_group_complete=run_group_complete,
            run_group_manifest_sha256=run_group_manifest_sha256,
            analysis_sha256=analysis_sha256,
            figure_sha256=figure_sha256,
            data_identity_sha256=data_identity_sha256,
            tokenizer_identity_sha256=tokenizer_identity_sha256,
            test_window_manifest_sha256=test_window_manifest_sha256,
            test_schedule_sha256=test_schedule_sha256,
        )
        opening_plan_sha256 = owned_sha256(
            "vfe4.wt103.test-opening-plan.v1",
            payload,
        )
        reservation_scope_sha256 = _opening_scope_sha256(
            endpoint_inventory=endpoint_inventory,
            run_group_manifest_sha256=run_group_manifest_sha256,
            data_identity_sha256=data_identity_sha256,
            tokenizer_identity_sha256=tokenizer_identity_sha256,
            test_window_manifest_sha256=test_window_manifest_sha256,
            test_schedule_sha256=test_schedule_sha256,
        )
        return cls(
            repository_root=repository_root,
            reservation_path=_canonical_reservation_path(
                repository_root,
                reservation_scope_sha256,
            ),
            durability_backend=durability_backend,
            endpoint_inventory=endpoint_inventory,
            terminal_checkpoints=terminal_checkpoints,
            run_group_complete=run_group_complete,
            run_group_manifest_sha256=run_group_manifest_sha256,
            analysis_sha256=analysis_sha256,
            figure_sha256=figure_sha256,
            data_identity_sha256=data_identity_sha256,
            tokenizer_identity_sha256=tokenizer_identity_sha256,
            test_window_manifest_sha256=test_window_manifest_sha256,
            test_schedule_sha256=test_schedule_sha256,
            reservation_scope_sha256=reservation_scope_sha256,
            opening_plan_sha256=opening_plan_sha256,
        )


@dataclass(frozen=True, init=False, slots=True)
class DurableTestOpeningCapability:
    schema_version: Literal["wt103-durable-test-opening-capability-v1"]
    opening_count: Literal[1]
    opening_plan_sha256: str
    endpoint_inventory_sha256: str
    estimator_protocol_sha256: str
    data_identity_sha256: str
    tokenizer_identity_sha256: str
    test_window_manifest_sha256: str
    test_schedule_sha256: str
    reservation_scope_sha256: str
    reservation_identity_sha256: str
    reservation_reopen_verified: Literal[True]
    capability_sha256: str

    def __new__(cls) -> "DurableTestOpeningCapability":
        raise TypeError(
            "DurableTestOpeningCapability can only be issued after reservation"
        )


@dataclass(slots=True)
class _IssuedCapability:
    capability: DurableTestOpeningCapability
    consumed: bool


_ISSUED_CAPABILITIES: dict[int, _IssuedCapability] = {}


def _validate_plan(plan: TestOpeningPlan) -> None:
    if type(plan) is not TestOpeningPlan:
        raise ValueError("plan must be an exact TestOpeningPlan")
    plan.__post_init__()
    if (
        not isinstance(plan.repository_root, Path)
        or not isinstance(plan.reservation_path, Path)
    ):
        raise ValueError("opening paths must be exact pathlib.Path values")
    root = plan.repository_root.resolve(strict=False)
    reservation = plan.reservation_path.resolve(strict=False)
    if not reservation.is_relative_to(root):
        raise ValueError("test opening reservation must remain inside the repository")
    canonical_reservation = _canonical_reservation_path(
        root,
        plan.reservation_scope_sha256,
    ).resolve(strict=False)
    if reservation != canonical_reservation:
        raise ValueError(
            "test opening requires the canonical reservation path"
        )
    if not isinstance(plan.durability_backend, DurabilityBackend):
        raise ValueError("opening plan requires an exact durability backend")
    if type(plan.endpoint_inventory) is not EndpointInventory:
        raise ValueError("opening plan requires an exact EndpointInventory")
    validate_endpoint_inventory(
        plan.endpoint_inventory,
        expected_sha256=plan.endpoint_inventory.endpoint_inventory_sha256,
    )
    if type(plan.run_group_complete) is not bool or not plan.run_group_complete:
        raise ValueError("test opening requires a complete run group")
    for name in (
        "run_group_manifest_sha256",
        "analysis_sha256",
        "figure_sha256",
        "data_identity_sha256",
        "tokenizer_identity_sha256",
        "test_window_manifest_sha256",
        "test_schedule_sha256",
        "reservation_scope_sha256",
        "opening_plan_sha256",
    ):
        _require_sha256(getattr(plan, name), name)
    if (
        type(plan.terminal_checkpoints) is not tuple
        or len(plan.terminal_checkpoints)
        != plan.endpoint_inventory.terminal_checkpoint_count
        or tuple(
            checkpoint.logical_key
            for checkpoint in plan.terminal_checkpoints
        )
        != plan.endpoint_inventory.terminal_checkpoint_keys
        or len(
            {
                checkpoint.logical_key
                for checkpoint in plan.terminal_checkpoints
            }
        )
        != len(plan.terminal_checkpoints)
    ):
        raise ValueError(
            "terminal checkpoint inventory differs from EndpointInventory"
        )
    for checkpoint in plan.terminal_checkpoints:
        if type(checkpoint) is not WT103CheckpointIdentity:
            raise ValueError("terminal checkpoint inventory has a wrong type")
        checkpoint.__post_init__()
        if checkpoint.checkpoint_role != "terminal_scoring":
            raise ValueError(
                "test opening accepts only terminal_scoring checkpoints"
            )


def _validate_durable_identity(
    identity: object,
    *,
    operation: Literal["exclusive_create"],
    payload: bytes,
) -> DurableFileIdentity:
    if type(identity) is not DurableFileIdentity:
        raise RuntimeError("durability backend returned an untyped identity")
    if (
        identity.operation != operation
        or identity.reopen_verified is not True
        or identity.size_bytes != len(payload)
    ):
        raise RuntimeError("durability backend did not reopen-verify the payload")
    expected = DurableFileIdentity.create(
        operation=operation,
        payload=payload,
        volume_identity=identity.volume_identity,
    )
    if identity != expected:
        raise RuntimeError(
            "durability backend identity does not authenticate reserved bytes"
        )
    return identity


def _issue_capability(
    *,
    plan: TestOpeningPlan,
    reservation: DurableFileIdentity,
) -> DurableTestOpeningCapability:
    payload = {
        "schema_version": "wt103-durable-test-opening-capability-v1",
        "opening_count": 1,
        "opening_plan_sha256": plan.opening_plan_sha256,
        "endpoint_inventory_sha256": (
            plan.endpoint_inventory.endpoint_inventory_sha256
        ),
        "estimator_protocol_sha256": (
            plan.endpoint_inventory.estimator_protocol_sha256
        ),
        "data_identity_sha256": plan.data_identity_sha256,
        "tokenizer_identity_sha256": plan.tokenizer_identity_sha256,
        "test_window_manifest_sha256": plan.test_window_manifest_sha256,
        "test_schedule_sha256": plan.test_schedule_sha256,
        "reservation_scope_sha256": plan.reservation_scope_sha256,
        "reservation_identity_sha256": reservation.identity_sha256,
        "reservation_reopen_verified": True,
    }
    capability = object.__new__(DurableTestOpeningCapability)
    for name, value in payload.items():
        object.__setattr__(capability, name, value)
    object.__setattr__(
        capability,
        "capability_sha256",
        owned_sha256(
            "vfe4.wt103.durable-test-opening-capability.v1",
            payload,
        ),
    )
    _ISSUED_CAPABILITIES[id(capability)] = _IssuedCapability(
        capability=capability,
        consumed=False,
    )
    return capability


def reserve_test_opening(
    plan: TestOpeningPlan,
) -> DurableTestOpeningCapability:
    """Reserve exactly one held-out transaction before issuing authority."""

    _validate_plan(plan)
    active_marker = plan.repository_root / ".verification" / "active.json"
    if os.path.lexists(active_marker):
        raise ValueError("active verification marker blocks test opening")
    reserved_payload = canonical_json_bytes_generic(
        {
            "schema_version": "wt103-test-opening-reservation-v1",
            "state": "RESERVED",
            "opening_count": 1,
            "opening_plan_sha256": plan.opening_plan_sha256,
            "endpoint_inventory_sha256": (
                plan.endpoint_inventory.endpoint_inventory_sha256
            ),
            "terminal_checkpoint_identity_sha256s": tuple(
                checkpoint.checkpoint_identity_sha256
                for checkpoint in plan.terminal_checkpoints
            ),
            "run_group_manifest_sha256": plan.run_group_manifest_sha256,
            "analysis_sha256": plan.analysis_sha256,
            "figure_sha256": plan.figure_sha256,
            "estimator_protocol_sha256": (
                plan.endpoint_inventory.estimator_protocol_sha256
            ),
            "data_identity_sha256": plan.data_identity_sha256,
            "tokenizer_identity_sha256": plan.tokenizer_identity_sha256,
            "test_window_manifest_sha256": plan.test_window_manifest_sha256,
            "test_schedule_sha256": plan.test_schedule_sha256,
            "reservation_scope_sha256": plan.reservation_scope_sha256,
        }
    )
    created_identity = plan.durability_backend.create_exclusive(
        plan.reservation_path,
        reserved_payload,
    )
    try:
        created = _validate_durable_identity(
            created_identity,
            operation="exclusive_create",
            payload=reserved_payload,
        )
    except Exception as exc:
        raise RuntimeError(
            "test opening is terminal after reservation; the immutable "
            "RESERVED record cannot be repaired or retried"
        ) from exc
    return _issue_capability(plan=plan, reservation=created)


def _require_issued_capability(
    capability: object,
) -> _IssuedCapability:
    if type(capability) is not DurableTestOpeningCapability:
        raise ValueError("test scoring requires an exact opening capability")
    issued = _ISSUED_CAPABILITIES.get(id(capability))
    if issued is None or issued.capability is not capability:
        raise ValueError("test opening capability is forged or no longer issued")
    return issued


def _unseal_test_windows(
    capability: DurableTestOpeningCapability,
    test_window_opener: Callable[[], CausalWindowSet],
) -> CausalWindowSet:
    """The sole private WT103 test-data unsealer."""

    issued = _require_issued_capability(capability)
    if issued.consumed:
        raise ValueError("test opening capability has already been consumed")
    if not callable(test_window_opener):
        raise ValueError("test unsealer requires a deferred window opener")
    issued.consumed = True
    sealed_test_windows = test_window_opener()
    if (
        type(sealed_test_windows) is not CausalWindowSet
        or sealed_test_windows.split != "test"
    ):
        raise ValueError("opening capability may unseal only exact test windows")
    sealed_test_windows.__post_init__()
    if (
        sealed_test_windows.manifest.manifest_sha256
        != capability.test_window_manifest_sha256
        or sealed_test_windows.tokenizer_spec.spec_sha256
        != capability.tokenizer_identity_sha256
        or sealed_test_windows.cache_record.raw_parent_sha256
        != capability.data_identity_sha256
    ):
        raise ValueError(
            "sealed test data/window/tokenizer identity differs from the opening"
        )
    return sealed_test_windows


__all__ = [
    "DurableTestOpeningCapability",
    "TestOpeningPlan",
    "reserve_test_opening",
]
