from __future__ import annotations

from dataclasses import replace

import pytest

from vfe4.training.checkpoint import (
    H6CheckpointManifest,
    load_h6_checkpoint,
    save_h6_checkpoint,
)
from vfe4.training.language import (
    H6AttemptCursor,
    H6AttemptSpec,
    H6ObjectiveManifest,
)
from vfe4.types import ArmId, H6ArmPhaseSchedule, TrainingPhase


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _exact_records() -> tuple[
    H6AttemptSpec,
    H6AttemptCursor,
    H6ObjectiveManifest,
]:
    schedule = H6ArmPhaseSchedule.create(
        endpoint_config_sha256=SHA_A,
        latent_enabled=False,
        phases=(TrainingPhase.MODEL_CE_ADAMW,),
    )
    attempt = H6AttemptSpec.create(
        source_git_head="1" * 40,
        dirty_digest=SHA_A,
        readiness_sha256=SHA_B,
        arm=ArmId.A0,
        config_id="h6-a0-ar-v1",
        endpoint_config_sha256=SHA_A,
        latent_enabled=False,
        objective_kind="cross_entropy",
        factory_sha256=SHA_B,
        model_family_sha256=SHA_C,
        training_authorization_sha256=SHA_D,
        objective_inventory_sha256=SHA_B,
        objective_adapter_sha256=SHA_C,
        h5_binding_sha256=SHA_D,
        outer_schedule_sha256=SHA_A,
        phase_schedule_sha256=schedule.phase_schedule_sha256,
        optimizer_policy_sha256=SHA_B,
        tuning_cell_sha256=SHA_C,
        training_seed=2026072301,
        data_identity_sha256=SHA_D,
        window_manifest_sha256=SHA_A,
        batch_schedule_sha256=SHA_B,
        estimator_sha256=SHA_C,
        prefix_certificate_sha256=SHA_D,
    )
    cursor = H6AttemptCursor.initial(
        attempt_spec=attempt,
        phase_schedule=schedule,
        data_cursor_sha256=SHA_A,
        rng_state_sha256=SHA_B,
    )
    objective = H6ObjectiveManifest.from_payload(
        {
            "attempt_spec_sha256": attempt.attempt_spec_sha256,
            "endpoint_config_sha256": SHA_A,
            "inventory_sha256": SHA_B,
            "adapter_sha256": SHA_C,
            "objective_kind": "cross_entropy",
            "producer_objective_sha256": SHA_D,
            "ordered_factor_bindings": (("emission", 1, SHA_A),),
            "total_raw_bytes_sha256": SHA_B,
            "detached_recognition_snapshot_sha256": None,
        }
    )
    return attempt, cursor, objective


def test_checkpoint_round_trip_is_byte_exact_and_rejects_tampering(
    tmp_path,
) -> None:
    attempt, cursor, objective = _exact_records()
    source = bytearray(b"\x00\x01tiny-model-state\xff")
    manifest = H6CheckpointManifest.capture(
        attempt_spec=attempt,
        cursor=cursor,
        objective_manifest=objective,
        active_state_bytes=(
            ("model.weight", bytes(source)),
            ("optimizer.model_ce_adamw.state", b"tiny-optimizer-state"),
        ),
    )
    first_path = tmp_path / "first.h6"
    save_h6_checkpoint(first_path, manifest)
    first_bytes = first_path.read_bytes()

    source[:] = b"x" * len(source)
    loaded = load_h6_checkpoint(
        first_path,
        expected_attempt_spec=attempt,
        expected_cursor=cursor,
        expected_objective_manifest=objective,
    )
    assert loaded.checkpoint_schema == "h6-checkpoint-v2"
    assert loaded.state_bytes("model.weight") == (
        b"\x00\x01tiny-model-state\xff"
    )

    second_path = tmp_path / "second.h6"
    save_h6_checkpoint(second_path, loaded)
    assert second_path.read_bytes() == first_bytes

    with pytest.raises(ValueError, match="schema"):
        replace(loaded, checkpoint_schema="h6-checkpoint-v0")

    corrupted = bytearray(first_bytes)
    corrupted[len(corrupted) // 2] ^= 0x01
    corrupt_path = tmp_path / "corrupt.h6"
    corrupt_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="integrity|digest|tamper"):
        load_h6_checkpoint(
            corrupt_path,
            expected_attempt_spec=attempt,
            expected_cursor=cursor,
            expected_objective_manifest=objective,
        )
