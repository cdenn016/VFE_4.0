from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from vfe4.artifacts.durability import (
    PosixDurabilityBackend,
    WindowsDurabilityBackend,
)
from vfe4.checkpoint import (
    MIGRATION_PROFILES,
    CheckpointError,
    CheckpointMigrationError,
    CheckpointSchemaError,
    CheckpointSecurityError,
    LoadedCheckpoint,
    MigrationProfile,
    ResumeContract,
    WT103CheckpointIdentity,
    load_checkpoint,
    make_checkpoint_identity,
    require_terminal_scoring,
    save_checkpoint,
    select_migration,
)
from vfe4.types.training import CheckpointBundle


_COMPATIBILITY_FIELDS = (
    "arm_spec_sha256",
    "experiment_plan_sha256",
    "config_sha256",
    "objective_sha256",
    "model_schema_sha256",
    "recognition_schema_sha256",
    "optimizer_schema_sha256",
    "scheduler_schema_sha256",
    "amp_schema_sha256",
    "rng_schema_sha256",
    "estimator_schema_sha256",
    "cursor_schema_sha256",
    "metric_schema_sha256",
    "update_trace_schema_sha256",
    "precision_profile_sha256",
    "dependency_lock_sha256",
    "source_sha256",
    "tokenizer_sha256",
    "data_sha256",
    "window_sha256",
    "permutation_sha256",
    "evidence_sha256",
    "environment_sha256",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _contract(
    *,
    checkpoint_role: str = "resume_only",
    training_complete: bool = False,
    **changes: object,
) -> ResumeContract:
    values: dict[str, object] = {
        "logical_key": "attempt/a5-parent/seed-2026072101/step-3",
        "checkpoint_role": checkpoint_role,
        "training_complete": training_complete,
        **{name: _sha(name) for name in _COMPATIBILITY_FIELDS},
        "maximum_checkpoint_bytes": 2 * 1024 * 1024,
        "maximum_tensor_bytes": 256 * 1024,
        "maximum_total_tensor_bytes": 512 * 1024,
        "maximum_tensor_count": 64,
        "maximum_container_items": 512,
        "maximum_recursion_depth": 16,
    }
    values.update(changes)
    return ResumeContract.create(**values)


def _scientific_state() -> dict[str, object]:
    return {
        "model_state": {
            "decoder.weight": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=torch.float32,
            )
        },
        "recognition_state": {
            "mean": torch.tensor([0.25, -0.5], dtype=torch.float64),
            "precision": torch.tensor([2.0, 3.0], dtype=torch.float64),
        },
        "optimizer_state": {
            "model": {
                "state": {
                    0: {
                        "step": torch.tensor(3, dtype=torch.int64),
                        "exp_avg": torch.tensor([0.1, 0.2]),
                        "exp_avg_sq": torch.tensor([0.01, 0.04]),
                    }
                },
                "param_groups": [
                    {
                        "lr": 3.0e-4,
                        "betas": (0.9, 0.999),
                        "eps": 1.0e-8,
                        "weight_decay": 0.01,
                    }
                ],
            }
        },
        "scheduler_state": {
            "model": {"last_epoch": 3, "step_count": 4, "base_lrs": [3.0e-4]}
        },
        "amp_scaler_state": {
            "scale": torch.tensor(65536.0),
            "growth_tracker": torch.tensor(7, dtype=torch.int32),
        },
        "rng_state": {
            "python": (3, (2147483648, 17, 29), None),
            "numpy": (
                "MT19937",
                torch.tensor([11, 13, 17], dtype=torch.int64),
                3,
                0,
                0.0,
            ),
            "torch_cpu": torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
            "torch_cuda": (torch.tensor([5, 6, 7, 8], dtype=torch.uint8),),
        },
        "estimator_state": {
            "stream_counters": {"validation": 5, "test": 0},
            "particle_level": 2,
        },
        "data_cursor_state": {
            "pass_index": 1,
            "batch_index": 3,
            "next_window_ids": (17, 18),
            "permutation_bytes": torch.tensor(
                [2, 0, 3, 1],
                dtype=torch.uint8,
            ),
            "permutation_sha256": _sha("permutation"),
        },
        "update_trace_state": {
            "global_step": 3,
            "successful_updates": 3,
            "rejected_updates": 1,
            "counted_targets": 96,
            "accepted_labels": ("recognition", "model"),
        },
        "metric_state": {
            "next_ordinal": 4,
            "hash_chain_head": _sha("metric-head"),
            "nll_numerator": 12.5,
            "nll_denominator": 3,
            "failure_ledger_head": _sha("failure-head"),
        },
        "next_prediction_fixture": (
            torch.tensor([[0.1, 0.9]], dtype=torch.float32),
            torch.tensor([[0.3, 0.7]], dtype=torch.float32),
        ),
    }


def _backend() -> PosixDurabilityBackend | WindowsDurabilityBackend:
    return WindowsDurabilityBackend() if os.name == "nt" else PosixDurabilityBackend()


class _FreshTarget:
    def __init__(self, contract: ResumeContract) -> None:
        self.checkpoint_contract_sha256 = contract.contract_sha256
        self.restored_state: dict[str, object] | None = None

    def is_fresh_checkpoint_target(self) -> bool:
        return self.restored_state is None

    def validate_checkpoint_state(self, state: dict[str, object]) -> None:
        assert set(state) == {
            "model_state",
            "recognition_state",
            "optimizer_state",
            "scheduler_state",
            "amp_scaler_state",
            "rng_state",
            "estimator_state",
            "data_cursor_state",
            "update_trace_state",
            "metric_state",
            "next_prediction_fixture",
        }
        assert self.restored_state is None

    def restore_checkpoint_state(self, state: dict[str, object]) -> None:
        assert self.restored_state is None
        self.restored_state = state


def _assert_scientific_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    if type(left) is torch.Tensor:
        assert isinstance(right, torch.Tensor)
        assert left.device.type == right.device.type == "cpu"
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
        return
    if type(left) is dict:
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_scientific_equal(left[key], right[key])
        return
    if type(left) in (list, tuple):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_scientific_equal(left_item, right_item)
        return
    assert left == right


def test_round_trip_restores_complete_state_and_separates_artifact_identity(
    tmp_path: Path,
) -> None:
    contract = _contract()
    state = _scientific_state()
    first_path = tmp_path / "resume-1.pt"
    second_path = tmp_path / "resume-2.pt"
    first = save_checkpoint(
        first_path,
        contract=contract,
        scientific_state=state,
        durability_backend=_backend(),
        operational_metadata={
            "process_id": 101,
            "utc_timestamp": "2026-07-28T01:02:03Z",
            "monotonic_seconds": 10.0,
            "elapsed_seconds": 1.0,
            "path_hint": "first",
            "write_ordinal": 1,
        },
    )
    second = save_checkpoint(
        second_path,
        contract=contract,
        scientific_state=state,
        durability_backend=_backend(),
        operational_metadata={
            "process_id": 202,
            "utc_timestamp": "2026-07-28T01:03:04Z",
            "monotonic_seconds": 20.0,
            "elapsed_seconds": 2.0,
            "path_hint": "second",
            "write_ordinal": 2,
        },
    )

    assert type(first) is WT103CheckpointIdentity
    assert first.checkpoint_role == "resume_only"
    assert first.scientific_state_sha256 == second.scientific_state_sha256
    assert first.checkpoint_payload_sha256 != second.checkpoint_payload_sha256
    assert first.checkpoint_manifest_body_sha256 != (
        second.checkpoint_manifest_body_sha256
    )
    assert first.artifact_sha256 != second.artifact_sha256
    assert (
        first.checkpoint_payload_sha256
        == hashlib.sha256(first_path.read_bytes()).hexdigest()
    )
    assert (
        first.artifact_sha256
        == hashlib.sha256(
            b"vfe4-checkpoint-artifact-v1\x00"
            + bytes.fromhex(first.checkpoint_payload_sha256)
            + bytes.fromhex(first.checkpoint_manifest_body_sha256)
        ).hexdigest()
    )

    target = _FreshTarget(contract)
    loaded = load_checkpoint(
        first_path,
        expected_identity=first,
        expected_contract=contract,
        fresh_target=target,
    )
    assert type(loaded) is LoadedCheckpoint
    assert type(loaded.bundle) is CheckpointBundle
    assert loaded.identity == first
    assert loaded.bundle.logical_key == contract.logical_key
    assert loaded.bundle.scientific_state_sha256 == (first.scientific_state_sha256)
    assert target.restored_state is not None
    _assert_scientific_equal(state, target.restored_state)

    for purpose in ("confirmation", "test", "endpoint", "figure"):
        with pytest.raises(CheckpointError, match="terminal_scoring"):
            require_terminal_scoring(first, purpose=purpose)

    terminal_contract = _contract(
        checkpoint_role="terminal_scoring",
        training_complete=True,
    )
    terminal = save_checkpoint(
        tmp_path / "terminal.pt",
        contract=terminal_contract,
        scientific_state=state,
        durability_backend=_backend(),
    )
    assert terminal.checkpoint_role == "terminal_scoring"
    assert terminal.scientific_state_sha256 == first.scientific_state_sha256
    require_terminal_scoring(terminal, purpose="test")
    with pytest.raises(CheckpointSchemaError, match="complete"):
        _contract(
            checkpoint_role="terminal_scoring",
            training_complete=False,
        )


def test_every_compatibility_field_blocks_before_fresh_target_mutation(
    tmp_path: Path,
) -> None:
    contract = _contract()
    path = tmp_path / "checkpoint.pt"
    identity = save_checkpoint(
        path,
        contract=contract,
        scientific_state=_scientific_state(),
        durability_backend=_backend(),
    )

    for field in _COMPATIBILITY_FIELDS:
        mismatch = _contract(**{field: _sha(f"changed-{field}")})
        target = _FreshTarget(mismatch)
        with pytest.raises(CheckpointError, match="compatibility|contract|mismatch"):
            load_checkpoint(
                path,
                expected_identity=identity,
                expected_contract=mismatch,
                fresh_target=target,
            )
        assert target.restored_state is None

    nonfresh = _FreshTarget(contract)
    nonfresh.restored_state = _scientific_state()
    with pytest.raises(CheckpointError, match="fresh"):
        load_checkpoint(
            path,
            expected_identity=identity,
            expected_contract=contract,
            fresh_target=nonfresh,
        )


def test_size_hash_corruption_and_inventory_tampering_fail_before_mutation(
    tmp_path: Path,
) -> None:
    contract = _contract()
    path = tmp_path / "checkpoint.pt"
    identity = save_checkpoint(
        path,
        contract=contract,
        scientific_state=_scientific_state(),
        durability_backend=_backend(),
    )
    payload = path.read_bytes()

    flipped = tmp_path / "flipped.pt"
    flipped.write_bytes(payload[:-1] + bytes((payload[-1] ^ 0x01,)))
    target = _FreshTarget(contract)
    with pytest.raises(CheckpointSecurityError, match="SHA-256"):
        load_checkpoint(
            flipped,
            expected_identity=identity,
            expected_contract=contract,
            fresh_target=target,
        )
    assert target.restored_state is None

    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(payload[:-16])
    with pytest.raises(CheckpointSecurityError, match="size"):
        load_checkpoint(
            truncated,
            expected_identity=identity,
            expected_contract=contract,
            fresh_target=_FreshTarget(contract),
        )

    wrong_size = make_checkpoint_identity(
        logical_key=identity.logical_key,
        checkpoint_role=identity.checkpoint_role,
        scientific_state_sha256=identity.scientific_state_sha256,
        checkpoint_payload_sha256=identity.checkpoint_payload_sha256,
        checkpoint_manifest_body_sha256=(identity.checkpoint_manifest_body_sha256),
        size_bytes=identity.size_bytes + 1,
    )
    with pytest.raises(CheckpointSecurityError, match="size"):
        load_checkpoint(
            path,
            expected_identity=wrong_size,
            expected_contract=contract,
            fresh_target=_FreshTarget(contract),
        )

    too_small = _contract(
        maximum_checkpoint_bytes=identity.size_bytes - 1,
        maximum_tensor_bytes=256,
        maximum_total_tensor_bytes=512,
    )
    with pytest.raises(CheckpointSecurityError, match="bound|maximum|size"):
        load_checkpoint(
            path,
            expected_identity=identity,
            expected_contract=too_small,
            fresh_target=_FreshTarget(too_small),
        )

    envelope = torch.load(
        io.BytesIO(payload),
        map_location="cpu",
        weights_only=True,
    )
    envelope["tensor_inventory"] = ()
    tampered_buffer = io.BytesIO()
    torch.save(envelope, tampered_buffer)
    tampered_payload = tampered_buffer.getvalue()
    inventory_tamper = tmp_path / "inventory-tamper.pt"
    inventory_tamper.write_bytes(tampered_payload)
    tampered_identity = make_checkpoint_identity(
        logical_key=identity.logical_key,
        checkpoint_role=identity.checkpoint_role,
        scientific_state_sha256=identity.scientific_state_sha256,
        checkpoint_payload_sha256=hashlib.sha256(tampered_payload).hexdigest(),
        checkpoint_manifest_body_sha256=(identity.checkpoint_manifest_body_sha256),
        size_bytes=len(tampered_payload),
    )
    target = _FreshTarget(contract)
    with pytest.raises(CheckpointSecurityError, match="inventory"):
        load_checkpoint(
            inventory_tamper,
            expected_identity=tampered_identity,
            expected_contract=contract,
            fresh_target=target,
        )
    assert target.restored_state is None


def test_nonregular_v3_paths_and_v3_keys_are_permanently_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    original = tmp_path / "checkpoint.pt"
    identity = save_checkpoint(
        original,
        contract=contract,
        scientific_state=_scientific_state(),
        durability_backend=_backend(),
    )

    redirect = tmp_path / "redirect.pt"
    redirect.write_bytes(original.read_bytes())
    real_is_junction = getattr(Path, "is_junction", None)

    def is_junction(path: Path) -> bool:
        if path == redirect:
            return True
        return bool(real_is_junction is not None and real_is_junction(path))

    monkeypatch.setattr(Path, "is_junction", is_junction, raising=False)
    with pytest.raises(CheckpointSecurityError, match="nonlink|reparse|junction"):
        load_checkpoint(
            redirect,
            expected_identity=identity,
            expected_contract=contract,
            fresh_target=_FreshTarget(contract),
        )

    v3_root = tmp_path / "V3_Transformer"
    v3_root.mkdir()
    with pytest.raises(CheckpointSecurityError, match="V3"):
        save_checkpoint(
            v3_root / "best_model.pt",
            contract=contract,
            scientific_state=_scientific_state(),
            durability_backend=_backend(),
        )
    with pytest.raises(CheckpointSchemaError, match="V3"):
        _contract(logical_key="v3_checkpoint/best_model.pt")


def _write_sentinel(path: str) -> int:
    Path(path).write_text("unsafe reducer executed", encoding="utf-8")
    return 1


class _MaliciousReducer:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (_write_sentinel, (str(self.sentinel),))


def test_weights_only_loader_never_executes_a_custom_reducer(
    tmp_path: Path,
) -> None:
    sentinel = tmp_path / "sentinel.txt"
    buffer = io.BytesIO()
    torch.save({"payload": _MaliciousReducer(sentinel)}, buffer)
    payload = buffer.getvalue()
    path = tmp_path / "malicious.pt"
    path.write_bytes(payload)
    contract = _contract()
    identity = make_checkpoint_identity(
        logical_key=contract.logical_key,
        checkpoint_role=contract.checkpoint_role,
        scientific_state_sha256=_sha("untrusted-scientific-state"),
        checkpoint_payload_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_manifest_body_sha256=_sha("untrusted-manifest-body"),
        size_bytes=len(payload),
    )
    target = _FreshTarget(contract)

    with pytest.raises(CheckpointSecurityError, match="weights-only|safe"):
        load_checkpoint(
            path,
            expected_identity=identity,
            expected_contract=contract,
            fresh_target=target,
        )
    assert not sentinel.exists()
    assert target.restored_state is None


def test_exact_whitelist_tensor_inventory_and_bounds_reject_before_write(
    tmp_path: Path,
) -> None:
    state = _scientific_state()
    backend = _backend()

    bad_extra = {**state, "terminal_run_manifest": "forbidden"}
    with pytest.raises(CheckpointSchemaError, match="scientific state|keys"):
        save_checkpoint(
            tmp_path / "extra.pt",
            contract=_contract(),
            scientific_state=bad_extra,
            durability_backend=backend,
        )

    bad_bytes = {
        **state,
        "estimator_state": {
            "stream_counters": {"validation": 0},
            "opaque_bytes": b"not-whitelisted",
        },
    }
    with pytest.raises(CheckpointSchemaError, match="unsupported|whitelist"):
        save_checkpoint(
            tmp_path / "bytes.pt",
            contract=_contract(),
            scientific_state=bad_bytes,
            durability_backend=backend,
        )

    incomplete_rng = {
        **state,
        "rng_state": {
            "torch_cpu": torch.tensor([1, 2], dtype=torch.uint8),
        },
    }
    with pytest.raises(CheckpointSchemaError, match="RNG|complete"):
        save_checkpoint(
            tmp_path / "incomplete-rng.pt",
            contract=_contract(),
            scientific_state=incomplete_rng,
            durability_backend=backend,
        )

    bad_parameter = {
        **state,
        "model_state": {
            "parameter": torch.nn.Parameter(torch.ones(1)),
        },
    }
    with pytest.raises(CheckpointSchemaError, match="exact|unsupported|Tensor"):
        save_checkpoint(
            tmp_path / "parameter.pt",
            contract=_contract(),
            scientific_state=bad_parameter,
            durability_backend=backend,
        )

    bad_dtype = {
        **state,
        "model_state": {
            "complex": torch.tensor([1.0 + 2.0j], dtype=torch.complex64),
        },
    }
    with pytest.raises(CheckpointSchemaError, match="dtype"):
        save_checkpoint(
            tmp_path / "complex.pt",
            contract=_contract(),
            scientific_state=bad_dtype,
            durability_backend=backend,
        )

    too_large = {
        **state,
        "model_state": {"large": torch.ones(3, dtype=torch.float32)},
    }
    with pytest.raises(CheckpointSchemaError, match="tensor.*bound|bytes"):
        save_checkpoint(
            tmp_path / "large.pt",
            contract=_contract(
                maximum_tensor_bytes=8,
                maximum_total_tensor_bytes=512 * 1024,
            ),
            scientific_state=too_large,
            durability_backend=backend,
        )

    assert not tuple(tmp_path.glob("*.pt"))


def test_manifest_body_has_no_run_manifest_circularity(
    tmp_path: Path,
) -> None:
    contract = _contract()
    state = _scientific_state()
    with pytest.raises(CheckpointSchemaError, match="run manifest|circular"):
        save_checkpoint(
            tmp_path / "circular-metadata.pt",
            contract=contract,
            scientific_state=state,
            durability_backend=_backend(),
            operational_metadata={
                "terminal_run_manifest_sha256": _sha("later-manifest"),
            },
        )
    circular_state = {
        **state,
        "metric_state": {
            **state["metric_state"],
            "run_manifest_sha256": _sha("later-manifest"),
        },
    }
    with pytest.raises(CheckpointSchemaError, match="run manifest|circular"):
        save_checkpoint(
            tmp_path / "circular-state.pt",
            contract=contract,
            scientific_state=circular_state,
            durability_backend=_backend(),
        )
    assert not tuple(tmp_path.iterdir())


def test_migration_registry_is_empty_and_v3_is_never_a_source_schema() -> None:
    assert dict(MIGRATION_PROFILES) == {}
    with pytest.raises(CheckpointMigrationError, match="V3.*never|permanent"):
        select_migration(
            source_schema="v3-checkpoint-v999",
            destination_schema="wt103-checkpoint-envelope-v1",
        )
    with pytest.raises(CheckpointMigrationError, match="no migration"):
        select_migration(
            source_schema="wt103-checkpoint-envelope-v0",
            destination_schema="wt103-checkpoint-envelope-v1",
        )
    with pytest.raises(CheckpointMigrationError, match="loss|test|hash"):
        MigrationProfile(
            source_schema_sha256=_sha("source"),
            destination_schema_sha256=_sha("destination"),
            transform_code_sha256=_sha("transform"),
            information_loss="unspecified",
            independent_test_sha256="",
            profile_sha256=_sha("invalid-profile"),
        )
