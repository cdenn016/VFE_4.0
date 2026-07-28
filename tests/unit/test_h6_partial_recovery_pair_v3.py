from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ("test_h6_training_attempt_v3",)


def test_checkpoint_first_partial_pair_reexecutes_and_completes_identically(
    _tiny_attempt_authority_v3: tuple[object, object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vfe4.training.h6_training_attempt_v3 as attempt_v3
    from vfe4.training.h6_execution_v3 import (
        bind_h6_executable_attempt_v3,
    )

    authorities, training_data, runtime = _tiny_attempt_authority_v3
    planned = next(
        attempt
        for attempt in authorities.plan.tuning_attempts  # type: ignore[union-attr]
        if attempt.endpoint_config_id == "h6-a0-transformer-v2"
    )
    executable = bind_h6_executable_attempt_v3(
        authorities=authorities,  # type: ignore[arg-type]
        planned_attempt=planned,
    )
    maximum_bytes = 256 * 1024 * 1024
    uninterrupted_path = (tmp_path / "a0-uninterrupted.h6v3").resolve()
    resumed_path = (tmp_path / "a0-checkpoint-first-resume.h6v3").resolve()
    uninterrupted = attempt_v3._execute_new_training_attempt_v3(
        executable=executable,
        training_data=training_data,
        runtime=runtime,
        checkpoint_path=uninterrupted_path,
        maximum_checkpoint_bytes=maximum_bytes,
    )
    uninterrupted_history = attempt_v3.read_h6_training_attempt_history_v3(
        checkpoint_path=uninterrupted_path,
        maximum_bytes=maximum_bytes,
    )

    recovery_checkpoint_path, recovery_history_path = (
        attempt_v3._recovery_artifact_paths_v3(
            checkpoint_path=resumed_path,
            ordinal=0,
            boundary_kind="terminal",
        )
    )
    progress_path = attempt_v3.h6_training_attempt_progress_path_v3(resumed_path)
    original_publish = attempt_v3._publish_immutable_bytes_v3
    interrupted = False

    def interrupt_before_first_history_publish(
        raw: bytes,
        *,
        path: Path,
        maximum_bytes: int,
        label: str,
    ) -> bytes:
        nonlocal interrupted
        if not interrupted and label == "attempt history shard":
            interrupted = True
            assert path == recovery_history_path
            assert recovery_checkpoint_path.is_file()
            assert not recovery_history_path.exists()
            raise RuntimeError("simulated checkpoint-first process loss")
        return original_publish(
            raw,
            path=path,
            maximum_bytes=maximum_bytes,
            label=label,
        )

    monkeypatch.setattr(
        attempt_v3,
        "_publish_immutable_bytes_v3",
        interrupt_before_first_history_publish,
    )
    with pytest.raises(RuntimeError, match="checkpoint-first process loss"):
        attempt_v3._execute_new_training_attempt_v3(
            executable=executable,
            training_data=training_data,
            runtime=runtime,
            checkpoint_path=resumed_path,
            maximum_checkpoint_bytes=maximum_bytes,
        )

    assert interrupted
    assert recovery_checkpoint_path.is_file()
    assert not recovery_history_path.exists()
    assert not progress_path.exists()
    assert not resumed_path.exists()
    partial_checkpoint_bytes = recovery_checkpoint_path.read_bytes()
    partial_checkpoint_mtime_ns = recovery_checkpoint_path.stat().st_mtime_ns
    assert (
        attempt_v3.recover_h6_training_attempt_v3(
            executable=executable,
            runtime=runtime,
            checkpoint_path=resumed_path,
            maximum_checkpoint_bytes=maximum_bytes,
        )
        is None
    )
    assert recovery_checkpoint_path.read_bytes() == partial_checkpoint_bytes
    assert recovery_checkpoint_path.stat().st_mtime_ns == partial_checkpoint_mtime_ns

    resumed = attempt_v3._execute_new_training_attempt_v3(
        executable=executable,
        training_data=training_data,
        runtime=runtime,
        checkpoint_path=resumed_path,
        maximum_checkpoint_bytes=maximum_bytes,
    )
    resumed_history = attempt_v3.read_h6_training_attempt_history_v3(
        checkpoint_path=resumed_path,
        maximum_bytes=maximum_bytes,
    )
    resumed_progress = attempt_v3.read_h6_training_attempt_progress_v3(
        path=progress_path,
        maximum_bytes=maximum_bytes,
    )

    assert recovery_checkpoint_path.read_bytes() == partial_checkpoint_bytes
    assert recovery_checkpoint_path.stat().st_mtime_ns == partial_checkpoint_mtime_ns
    assert recovery_history_path.is_file()
    assert tuple(
        (boundary.ordinal, boundary.boundary_kind)
        for boundary in resumed_progress.boundaries
    ) == ((0, "terminal"),)
    assert resumed.terminal_cursor.recognition_update_count == 0
    assert resumed.terminal_cursor.model_update_count == 1
    assert len(resumed_history.metric_history) == 1
    assert len(resumed_history.validation_boundary_history) == 1
    assert resumed.checkpoint_bytes_sha256 == uninterrupted.checkpoint_bytes_sha256
    assert resumed_path.read_bytes() == uninterrupted_path.read_bytes()
    assert resumed_history == uninterrupted_history
