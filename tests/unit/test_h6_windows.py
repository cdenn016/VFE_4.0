from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from vfe4.data.byte_tokenizer import BOS_ID, ByteTokenizerV1
from vfe4.data.windows import (
    CausalPrefix,
    VALIDATION_SAFETY_FIXTURE_COUNT,
    build_causal_windows,
    evaluation_batches,
    frozen_batch_schedule,
    materialize_validation_safety_fixture,
    quarter_pass_batches,
    schedule_batches,
)


def _independent_permutation(n: int, pass_index: int) -> tuple[int, ...]:
    values = list(range(n))
    words: list[int] = []
    counter = 0
    for i in range(n - 1, 0, -1):
        modulus = i + 1
        limit = (1 << 64) - ((1 << 64) % modulus)
        while True:
            if not words:
                digest = hashlib.sha256(
                    b"VFE4-H6-BATCH-PERMUTATION-DRAW-V1\x00"
                    + (2026072199).to_bytes(8, "little")
                    + pass_index.to_bytes(8, "little")
                    + counter.to_bytes(8, "little")
                ).digest()
                words = [
                    int.from_bytes(digest[offset : offset + 8], "little")
                    for offset in (0, 8, 16, 24)
                ]
                counter += 1
            draw = words.pop(0)
            if draw < limit:
                break
        j = draw % modulus
        values[i], values[j] = values[j], values[i]
    return tuple(values)


def test_causal_windows_use_exact_stride_padding_and_target_counts() -> None:
    tokens = tuple(range(35))
    windows = build_causal_windows(tokens, split="validation")

    assert windows.starts == (0, 32)
    assert windows.inputs[0] == tuple(range(32))
    assert windows.targets[0] == tuple(range(1, 33))
    assert windows.inputs[1][:3] == (32, 33, 34)
    assert windows.inputs[1][3:] == (BOS_ID,) * 29
    assert windows.targets[1][:2] == (33, 34)
    assert windows.targets[1][2:] == (-100,) * 30
    assert windows.real_target_counts == (32, 2)
    assert windows.counted_target_total == 34
    assert windows.attention_masks[1] == (True, True, True) + (False,) * 29


def test_causal_prefix_is_target_free_owned_and_hash_bound() -> None:
    vocabulary = ByteTokenizerV1().vocabulary_identity
    source = torch.tensor([256, 65], dtype=torch.int64, device="cpu")
    prefix = CausalPrefix.create(
        receiver_t=3,
        vocabulary=vocabulary,
        token_ids=source,
    )
    source[0] = 0
    exposed = prefix.token_ids
    exposed[1] = 0

    assert prefix.receiver_t == 3
    assert torch.equal(prefix.token_ids, torch.tensor([256, 65], dtype=torch.int64))
    assert not hasattr(prefix, "target_ids")
    assert not hasattr(prefix, "attention_mask")
    with pytest.raises(FrozenInstanceError):
        prefix.receiver_t = 4  # type: ignore[misc]

    invalid = (
        (0, torch.empty(0, dtype=torch.int64)),
        (3, torch.tensor([256], dtype=torch.int64)),
        (2, torch.tensor([258], dtype=torch.int64)),
        (2, torch.tensor([1.0], dtype=torch.float64)),
        (2, torch.tensor([[1]], dtype=torch.int64)),
    )
    for receiver_t, token_ids in invalid:
        with pytest.raises(ValueError):
            CausalPrefix.create(
                receiver_t=receiver_t,
                vocabulary=vocabulary,
                token_ids=token_ids,
            )


def test_causal_windows_build_target_free_prefix_for_one_receiver() -> None:
    windows = build_causal_windows(tuple(range(35)), split="train")
    vocabulary = ByteTokenizerV1().vocabulary_identity
    prefix = windows.causal_prefix(
        window_index=0,
        receiver_t=5,
        vocabulary=vocabulary,
    )
    assert torch.equal(prefix.token_ids, torch.tensor([0, 1, 2, 3]))
    assert prefix.receiver_t == 5


def test_validation_fixture_has_exact_ranked_binary_preimage() -> None:
    raw = bytes(range(256)) * 513
    tokens = ByteTokenizerV1().encode(raw)
    storage = ByteTokenizerV1().storage_identity(tokens)

    fixture = materialize_validation_safety_fixture(
        validation_tokens=tokens,
        validation_storage_identity=storage,
    )

    all_windows = build_causal_windows(tokens, split="validation")
    token_digest = bytes.fromhex(storage.encoded_token_sha256)
    ranked = sorted(
        range(len(all_windows.starts)),
        key=lambda index: (
            hashlib.sha256(
                b"VFE4-H6-VALIDATION-SAFETY-RANK-V1\x00"
                + token_digest
                + index.to_bytes(8, "little")
            ).digest(),
            index,
        ),
    )[:VALIDATION_SAFETY_FIXTURE_COUNT]
    expected = bytearray(
        b"VFE4-H6-VALIDATION-SAFETY-FIXTURE-V1\x00"
        + token_digest
        + VALIDATION_SAFETY_FIXTURE_COUNT.to_bytes(4, "little")
    )
    for index in ranked:
        start = all_windows.starts[index]
        real_count = all_windows.real_target_counts[index]
        row = tuple(tokens[start : start + 33])
        row += (BOS_ID,) * (33 - len(row))
        expected += start.to_bytes(8, "little")
        expected += real_count.to_bytes(2, "little")
        expected += b"".join(value.to_bytes(2, "little") for value in row)

    assert fixture.starts == tuple(all_windows.starts[index] for index in ranked)
    assert fixture.real_target_counts == tuple(
        all_windows.real_target_counts[index] for index in ranked
    )
    assert fixture.fixture_sha256 == hashlib.sha256(expected).hexdigest()
    fixture.verify_fixture_bytes(bytes(expected))


def test_validation_fixture_rejects_too_few_or_mismatched_tokens() -> None:
    tokenizer = ByteTokenizerV1()
    tokens = tokenizer.encode(b"short")
    storage = tokenizer.storage_identity(tokens)
    with pytest.raises(ValueError, match="4,096"):
        materialize_validation_safety_fixture(
            validation_tokens=tokens,
            validation_storage_identity=storage,
        )
    with pytest.raises(ValueError, match="identity"):
        materialize_validation_safety_fixture(
            validation_tokens=tokenizer.encode(bytes(range(256)) * 513),
            validation_storage_identity=storage,
        )


def test_frozen_schedule_matches_counter_fisher_yates_and_batch_policies() -> None:
    schedule = frozen_batch_schedule(window_count=19, zero_based_pass_index=3)
    expected = _independent_permutation(19, 3)

    assert schedule.permutation == expected
    batches = schedule_batches(schedule)
    assert tuple(value for batch in batches for value in batch) == expected
    assert tuple(map(len, batches)) == (8, 8, 3)
    assert quarter_pass_batches(schedule) == batches[: math.ceil(len(batches) / 4)]
    assert evaluation_batches(19) == (
        tuple(range(8)),
        tuple(range(8, 16)),
        tuple(range(16, 19)),
    )


def test_window_and_schedule_inputs_fail_closed() -> None:
    for tokens in ((), (1,), (0, 258), (0, -1)):
        with pytest.raises(ValueError):
            build_causal_windows(tokens, split="train")
    for window_count, pass_index in ((0, 0), (1, -1)):
        with pytest.raises(ValueError):
            frozen_batch_schedule(
                window_count=window_count,
                zero_based_pass_index=pass_index,
            )
