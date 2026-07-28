from __future__ import annotations

import dataclasses
import hashlib
import inspect
import sys
from pathlib import Path

import pytest
import torch


class _ByteAdapter:
    distribution_name = "tiktoken"
    distribution_version = "0.12.0"
    encoding_name = "gpt2"
    vocabulary_size = 50_257
    special_tokens = (("<|endoftext|>", 50_256),)
    regex_pattern_sha256 = hashlib.sha256(b"window-regex").hexdigest()
    mergeable_ranks_sha256 = hashlib.sha256(b"window-ranks").hexdigest()
    ordinary_encoding_policy = "encode_ordinary_no_special_tokens"
    fitted_state_sha256 = None
    implementation_sha256 = hashlib.sha256(
        b"tests.wt103.window-byte-adapter.v1"
    ).hexdigest()

    def encode_ordinary(self, text: str) -> tuple[int, ...]:
        return tuple(text.encode("utf-8"))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return bytes(token_ids).decode("utf-8")


class _Backend:
    def publish_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def _fixture_contract():
    from vfe4.data.tokenizer import SyntheticTokenizerFixtureContract

    adapter = _ByteAdapter()
    return SyntheticTokenizerFixtureContract.create(
        distribution_name="tiktoken",
        distribution_version="0.12.0",
        encoding_name="gpt2",
        vocabulary_size=50_257,
        special_tokens=adapter.special_tokens,
        regex_pattern_sha256=adapter.regex_pattern_sha256,
        mergeable_ranks_sha256=adapter.mergeable_ranks_sha256,
        ordinary_encoding_policy=adapter.ordinary_encoding_policy,
        golden_vectors=(
            ("ascii", "window\n", adapter.encode_ordinary("window\n")),
            ("unicode", "π\n", adapter.encode_ordinary("π\n")),
            ("newlines", "\n\n", adapter.encode_ordinary("\n\n")),
        ),
    )


def _cache(tmp_path: Path, *, split: str, raw: bytes):
    from vfe4.data.tokenizer import (
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
    )

    adapter = _ByteAdapter()
    contract = _fixture_contract()
    spec = build_synthetic_fixture_tokenizer_spec(contract, adapter)
    record = encode_fixture_split_record(
        split=split,
        raw_bytes=raw,
        raw_parent_sha256=hashlib.sha256(split.encode() + raw).hexdigest(),
        spec=spec,
        fixture_contract=contract,
        adapter=adapter,
        cache_root=tmp_path / "cache",
        durability_backend=_Backend(),
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    return spec, record


def test_real_durability_backend_provisions_window_and_schedule_trees(
    tmp_path: Path,
) -> None:
    from vfe4.artifacts.durability import (
        PosixDurabilityBackend,
        WindowsDurabilityBackend,
    )
    from vfe4.data.tokenizer import (
        build_synthetic_fixture_tokenizer_spec,
        encode_fixture_split_record,
        issue_fixture_split_capability,
    )
    from vfe4.data.windows import (
        build_train_schedule,
        materialize_causal_window_set,
    )

    backend = (
        WindowsDurabilityBackend()
        if sys.platform == "win32"
        else PosixDurabilityBackend()
    )
    adapter = _ByteAdapter()
    contract = _fixture_contract()
    spec = build_synthetic_fixture_tokenizer_spec(contract, adapter)
    raw = bytes((index % 95) + 32 for index in range(300))
    record = encode_fixture_split_record(
        split="train",
        raw_bytes=raw,
        raw_parent_sha256=hashlib.sha256(raw).hexdigest(),
        spec=spec,
        fixture_contract=contract,
        adapter=adapter,
        cache_root=tmp_path / "cache",
        durability_backend=backend,
        available_disk_bytes=20 * 1024**3,
        available_host_ram_bytes=20 * 1024**3,
        smoke_token_bytes_per_raw_byte=4.0,
        smoke_ram_bytes_per_raw_byte=2.0,
    )
    split_capability = issue_fixture_split_capability(
        allowed_splits=("train",),
        cache_identities=(record,),
    )
    windows = materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=split_capability,
        artifact_root=tmp_path / "windows",
        durability_backend=backend,
    )
    schedule = build_train_schedule(
        windows=windows,
        pass_index=0,
        artifact_root=tmp_path / "schedules",
        durability_backend=backend,
        batch_size=2,
    )

    assert (tmp_path / "windows" / windows.row_payload_relative_path).is_file()
    permutation = schedule.permutation_manifest
    assert permutation is not None
    assert tuple(schedule.window_ids) != ()
    assert any((tmp_path / "schedules").rglob("*.u64le"))


def test_owned_artifact_paths_reject_reparse_components_without_following(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe4.artifacts.paths import (
        OwnedArtifactPathError,
        owned_payload_path,
    )

    root = tmp_path / "artifacts"
    root.mkdir()
    redirected = root / "redirected"
    redirected.mkdir()
    path_type = type(redirected)
    monkeypatch.setattr(
        path_type,
        "is_junction",
        lambda self: self == redirected,
    )

    with pytest.raises(
        OwnedArtifactPathError,
        match="nonlink|symlink|junction|reparse",
    ):
        owned_payload_path(
            root=root,
            relative_path="redirected/payload.bin",
            prepare_parents=False,
        )


def test_exhaustive_window_rows_cover_every_transition_exactly_once() -> None:
    from vfe4.data.windows import enumerate_wt103_window_rows

    for token_count in range(2, 521):
        rows = enumerate_wt103_window_rows(token_count)
        expected_transition_ids = tuple(range(token_count - 1))
        observed_transition_ids = tuple(
            transition_id
            for row in rows
            for transition_id in range(
                row.start_transition,
                row.start_transition + row.counted_targets,
            )
        )
        assert observed_transition_ids == expected_transition_ids
        assert sum(row.counted_targets for row in rows) == token_count - 1
        assert rows[-1].counted_targets <= 128
        assert len(rows) == (token_count - 2) // 128 + 1


@pytest.mark.parametrize("token_count", (2, 128, 129, 130, 257, 520))
def test_materialized_windows_use_eot_padding_and_minus100_targets(
    tmp_path: Path, token_count: int
) -> None:
    from vfe4.data.tokenizer import issue_fixture_split_capability
    from vfe4.data.windows import materialize_causal_window_set

    raw = bytes((index % 95) + 32 for index in range(token_count))
    spec, record = _cache(tmp_path, split="train", raw=raw)
    capability = issue_fixture_split_capability(
        allowed_splits=("train",),
        cache_identities=(record,),
    )
    windows = materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_Backend(),
    )

    assert windows.manifest.counted_targets == token_count - 1
    assert windows.manifest.window_count == len(windows.rows)
    observed: list[tuple[int, int]] = []
    for window_id in range(windows.manifest.window_count):
        window = windows.window(window_id)
        assert window.inputs.dtype is torch.int64
        assert window.targets.dtype is torch.int64
        assert window.attention_mask.dtype is torch.bool
        assert tuple(window.inputs.shape) == (128,)
        assert tuple(window.targets.shape) == (128,)
        assert tuple(window.attention_mask.shape) == (128,)
        real = window.counted_targets
        assert bool(torch.all(window.attention_mask[:real]))
        assert not bool(torch.any(window.attention_mask[real:]))
        assert bool(torch.all(window.inputs[real:] == 50_256))
        assert bool(torch.all(window.targets[real:] == -100))
        for input_id, target_id in zip(
            window.inputs[:real].tolist(),
            window.targets[:real].tolist(),
            strict=True,
        ):
            observed.append((input_id, target_id))
    assert observed == list(zip(raw[:-1], raw[1:], strict=True))


def test_train_permutation_is_stored_stable_complete_and_pass_specific(
    tmp_path: Path,
) -> None:
    from vfe4.data.tokenizer import issue_fixture_split_capability
    from vfe4.data.windows import (
        build_train_schedule,
        iter_causal_batches,
        materialize_causal_window_set,
    )

    spec, record = _cache(tmp_path, split="train", raw=b"x" * 520)
    capability = issue_fixture_split_capability(
        allowed_splits=("train",),
        cache_identities=(record,),
    )
    windows = materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_Backend(),
    )
    schedule_a = build_train_schedule(
        windows=windows,
        pass_index=0,
        artifact_root=tmp_path / "schedules-a",
        durability_backend=_Backend(),
    )
    schedule_b = build_train_schedule(
        windows=windows,
        pass_index=0,
        artifact_root=tmp_path / "schedules-b",
        durability_backend=_Backend(),
    )
    schedule_pass_1 = build_train_schedule(
        windows=windows,
        pass_index=1,
        artifact_root=tmp_path / "schedules-c",
        durability_backend=_Backend(),
    )

    assert schedule_a.window_ids == schedule_b.window_ids
    assert schedule_a.schedule_sha256 == schedule_b.schedule_sha256
    assert schedule_a.permutation_manifest == schedule_b.permutation_manifest
    assert schedule_a.window_ids != schedule_pass_1.window_ids
    assert sorted(schedule_a.window_ids) == list(
        range(windows.manifest.window_count)
    )
    batches = tuple(iter_causal_batches(windows=windows, schedule=schedule_a))
    assert sum(len(batch.window_ids) for batch in batches) == len(windows.rows)
    assert batches[-1].window_ids


@pytest.mark.parametrize("split", ("validation", "test"))
def test_evaluation_schedules_are_strictly_ascending(
    tmp_path: Path, split: str
) -> None:
    from vfe4.data.tokenizer import issue_fixture_split_capability
    from vfe4.data.windows import (
        build_evaluation_schedule,
        materialize_causal_window_set,
    )

    spec, record = _cache(tmp_path, split=split, raw=b"y" * 300)
    allowed = ("validation",) if split == "validation" else ("train",)
    # Task 4 deliberately cannot issue a test-reading capability.
    if split == "test":
        with pytest.raises(ValueError, match="test|allowed"):
            issue_fixture_split_capability(
                allowed_splits=allowed,
                cache_identities=(record,),
            )
        return
    capability = issue_fixture_split_capability(
        allowed_splits=allowed,
        cache_identities=(record,),
    )
    windows = materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_Backend(),
    )
    schedule = build_evaluation_schedule(windows)
    assert schedule.window_ids == tuple(range(windows.manifest.window_count))


def test_cursor_resume_matches_the_exact_next_batch_and_denominator(
    tmp_path: Path,
) -> None:
    from vfe4.data.tokenizer import issue_fixture_split_capability
    from vfe4.data.windows import (
        build_train_schedule,
        cursor_after_batches,
        iter_causal_batches,
        materialize_causal_window_set,
    )

    spec, record = _cache(tmp_path, split="train", raw=b"z" * 520)
    capability = issue_fixture_split_capability(
        allowed_splits=("train",),
        cache_identities=(record,),
    )
    windows = materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_Backend(),
    )
    schedule = build_train_schedule(
        windows=windows,
        pass_index=0,
        artifact_root=tmp_path / "schedules",
        durability_backend=_Backend(),
        batch_size=2,
    )
    uninterrupted = tuple(
        iter_causal_batches(
            windows=windows,
            schedule=schedule,
            batch_size=2,
        )
    )
    for boundary in range(len(uninterrupted) + 1):
        cursor = cursor_after_batches(
            windows=windows,
            schedule=schedule,
            completed_batch_count=boundary,
            batch_size=2,
        )
        resumed = tuple(
            iter_causal_batches(
                windows=windows,
                schedule=schedule,
                batch_size=2,
                cursor=cursor,
            )
        )
        assert tuple(batch.window_ids for batch in resumed) == tuple(
            batch.window_ids for batch in uninterrupted[boundary:]
        )
        assert cursor.counted_targets == sum(
            batch.counted_targets for batch in uninterrupted[:boundary]
        )
        expected_next = (
            uninterrupted[boundary].window_ids
            if boundary < len(uninterrupted)
            else ()
        )
        assert cursor.next_window_ids == expected_next


def test_cursor_and_permutation_tampering_fail_before_consumption(
    tmp_path: Path,
) -> None:
    from vfe4.data.tokenizer import issue_fixture_split_capability
    from vfe4.data.windows import (
        build_train_schedule,
        cursor_after_batches,
        iter_causal_batches,
        materialize_causal_window_set,
    )

    spec, record = _cache(tmp_path, split="train", raw=b"a" * 300)
    capability = issue_fixture_split_capability(
        allowed_splits=("train",),
        cache_identities=(record,),
    )
    windows = materialize_causal_window_set(
        cache_record=record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_Backend(),
    )
    schedule = build_train_schedule(
        windows=windows,
        pass_index=0,
        artifact_root=tmp_path / "schedules",
        durability_backend=_Backend(),
    )
    cursor = cursor_after_batches(
        windows=windows,
        schedule=schedule,
        completed_batch_count=0,
    )
    with pytest.raises(ValueError, match="cursor|sha256|next"):
        bad_cursor = dataclasses.replace(
            cursor, permutation_sha256="0" * 64
        )
        tuple(
            iter_causal_batches(
                windows=windows,
                schedule=schedule,
                cursor=bad_cursor,
            )
        )
    with pytest.raises(ValueError, match="schedule|permutation|duplicate"):
        bad_schedule = dataclasses.replace(
            schedule,
            window_ids=(0,) * len(schedule.window_ids),
        )
        tuple(iter_causal_batches(windows=windows, schedule=bad_schedule))


def test_wt103_training_access_has_no_test_unsealing_path() -> None:
    import vfe4.data.access as access

    assert hasattr(access, "WT103TrainDataCapability")
    assert hasattr(access, "issue_synthetic_wt103_train_capability")
    assert not hasattr(access, "open_wt103_test")
    assert not hasattr(access, "unseal_wt103_test")
    source = inspect.getsource(access.issue_synthetic_wt103_train_capability)
    assert "test" not in source.casefold()


def test_train_validation_materialization_requires_both_bound_records(
    tmp_path: Path,
) -> None:
    from vfe4.data.access import (
        issue_synthetic_wt103_train_capability,
        materialize_train_data,
    )
    from vfe4.data.tokenizer import issue_fixture_split_capability
    from vfe4.data.windows import materialize_causal_window_set

    spec, train_record = _cache(tmp_path, split="train", raw=b"train")
    validation_spec, validation_record = _cache(
        tmp_path, split="validation", raw=b"valid"
    )
    assert validation_spec == spec
    split_capability = issue_fixture_split_capability(
        allowed_splits=("train", "validation"),
        cache_identities=(train_record, validation_record),
    )
    authority = issue_synthetic_wt103_train_capability(
        split_capability=split_capability,
        tokenizer_spec=spec,
        train_cache=train_record,
        validation_cache=validation_record,
    )
    train_windows = materialize_causal_window_set(
        cache_record=train_record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=split_capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_Backend(),
    )
    validation_windows = materialize_causal_window_set(
        cache_record=validation_record,
        tokenizer_spec=spec,
        cache_root=tmp_path / "cache",
        split_capability=split_capability,
        artifact_root=tmp_path / "windows",
        durability_backend=_Backend(),
    )
    training_data = materialize_train_data(
        capability=authority,
        train_windows=train_windows,
        validation_windows=validation_windows,
    )
    assert training_data.train is train_windows
    assert training_data.validation is validation_windows
    with pytest.raises(ValueError, match="authority|match"):
        materialize_train_data(
            capability=authority,
            train_windows=train_windows,
            validation_windows=train_windows,
        )
