from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch

from vfe4.inference.h4_instrumentation import (
    CountingOperationRecorder,
    InstrumentedLinearAlgebra,
    NullOperationRecorder,
    measure_untimed_memory,
)
from vfe4.types.h4 import H4MemoryRecord, H4OperationRecord


def test_public_instrumentation_signatures_and_recorder_surface_are_frozen() -> None:
    assert tuple(inspect.signature(InstrumentedLinearAlgebra).parameters) == (
        "problem_id",
        "arm",
        "recorder",
    )
    assert tuple(inspect.signature(InstrumentedLinearAlgebra.cholesky).parameters) == (
        "self",
        "value",
    )
    assert tuple(
        inspect.signature(InstrumentedLinearAlgebra.triangular_solve).parameters
    ) == ("self", "triangular", "rhs", "upper")
    assert tuple(
        inspect.signature(InstrumentedLinearAlgebra.matrix_multiply).parameters
    ) == ("self", "left", "right")
    assert tuple(
        inspect.signature(InstrumentedLinearAlgebra.symmetric_rank_update).parameters
    ) == ("self", "covariance", "gain", "innovation_covariance")
    assert tuple(
        inspect.signature(InstrumentedLinearAlgebra.selected_block_extract).parameters
    ) == ("self", "value", "row_indices", "column_indices")
    assert tuple(inspect.signature(measure_untimed_memory).parameters) == (
        "problem_id",
        "arm",
        "callable",
    )

    for recorder in (NullOperationRecorder(), CountingOperationRecorder()):
        public = {name for name in dir(recorder) if not name.startswith("_")}
        assert public == {"snapshot"}
        assert recorder.snapshot() == ()
        for forbidden in (
            "record",
            "observe",
            "increment",
            "record_only",
            "capability",
        ):
            assert not hasattr(recorder, forbidden)


def test_counting_facade_records_real_successful_operations_once() -> None:
    recorder = CountingOperationRecorder()
    linalg = InstrumentedLinearAlgebra(
        problem_id="problem-a",
        arm="information",
        recorder=recorder,
    )
    left = torch.tensor(((1.0, 2.0), (3.0, 4.0)), dtype=torch.float64)
    right = torch.tensor(((2.0, 0.0), (0.0, 2.0)), dtype=torch.float64)

    expected = left @ right
    assert torch.equal(linalg.matrix_multiply(left, right), expected)
    assert torch.equal(linalg.matrix_multiply(left, right), expected)
    assert torch.equal(linalg.cholesky(torch.eye(2, dtype=torch.float64)), torch.eye(2, dtype=torch.float64))

    assert recorder.snapshot() == (
        H4OperationRecord(
            "problem-a",
            "information",
            "matrix_multiply",
            ((2, 2), (2, 2)),
            (2, 2),
            2,
        ),
        H4OperationRecord(
            "problem-a",
            "information",
            "cholesky",
            ((2, 2),),
            (2, 2),
            1,
        ),
    )

    before = recorder.snapshot()
    with pytest.raises(RuntimeError):
        linalg.cholesky(torch.tensor(((1.0, 2.0), (2.0, 1.0)), dtype=torch.float64))
    assert recorder.snapshot() == before


def test_null_and_counting_facades_are_numerically_identical_and_identity_bound() -> None:
    null = InstrumentedLinearAlgebra(
        problem_id="problem-b",
        arm="moment",
        recorder=NullOperationRecorder(),
    )
    counting_recorder = CountingOperationRecorder()
    counting = InstrumentedLinearAlgebra(
        problem_id="problem-b",
        arm="moment",
        recorder=counting_recorder,
    )
    covariance = torch.tensor(((2.0, 0.3), (0.3, 1.5)), dtype=torch.float64)
    gain = torch.tensor(((0.2,), (0.4,)), dtype=torch.float64)
    innovation = torch.tensor(((1.25,),), dtype=torch.float64)
    vector = torch.tensor((4.0, 5.0, 6.0), dtype=torch.float64)
    matrix = torch.arange(16, dtype=torch.float64).reshape(4, 4)

    operations = (
        lambda facade: facade.triangular_solve(
            torch.tensor(((2.0, 0.0), (1.0, 3.0)), dtype=torch.float64),
            torch.tensor((2.0, 7.0), dtype=torch.float64),
        ),
        lambda facade: facade.symmetric_rank_update(covariance, gain, innovation),
        lambda facade: facade.selected_block_extract(vector, (0, 2)),
        lambda facade: facade.selected_block_extract(matrix, (0, 3), (1, 2)),
    )
    for operation in operations:
        assert torch.equal(operation(null), operation(counting))

    assert null.problem_id == counting.problem_id == "problem-b"
    assert null.arm == counting.arm == "moment"
    with pytest.raises((AttributeError, FrozenInstanceError)):
        null.problem_id = "spoofed"  # type: ignore[misc]
    with pytest.raises((AttributeError, FrozenInstanceError)):
        null.arm = "information"  # type: ignore[misc]
    assert counting_recorder.snapshot()


def test_selected_extraction_and_symmetric_rank_update_are_real_operations() -> None:
    recorder = CountingOperationRecorder()
    linalg = InstrumentedLinearAlgebra(
        problem_id="problem-c",
        arm="moment",
        recorder=recorder,
    )
    value = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    assert torch.equal(
        linalg.selected_block_extract(value, (3, 1), (4, 0, 2)),
        value[torch.tensor((3, 1))[:, None], torch.tensor((4, 0, 2))[None, :]],
    )
    with pytest.raises(ValueError):
        linalg.selected_block_extract(value, (0, 0), (1,))

    covariance = torch.tensor(((2.0, 0.2), (0.2, 1.0)), dtype=torch.float64)
    gain = torch.tensor(((0.5,), (0.25,)), dtype=torch.float64)
    innovation = torch.tensor(((0.8,),), dtype=torch.float64)
    updated = linalg.symmetric_rank_update(covariance, gain, innovation)
    expected = covariance - gain @ innovation @ gain.T
    assert torch.allclose(updated, expected, rtol=0.0, atol=0.0)
    assert torch.equal(updated, updated.T)


def test_untimed_memory_measurement_calls_work_once_and_fails_closed() -> None:
    calls: list[int] = []

    def work() -> object:
        calls.append(1)
        return [0] * 128

    record = measure_untimed_memory("problem-d", "information", work)
    assert type(record) is H4MemoryRecord
    assert record.problem_id == "problem-d"
    assert record.arm == "information"
    assert calls == [1]
    assert record.python_peak_bytes is not None
    assert record.python_peak_bytes >= 0
    assert record.process_working_set_delta_bytes is None
    assert record.unavailable_fields == ("process_working_set_delta_bytes",)

    def fail() -> object:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        measure_untimed_memory("problem-d", "information", fail)
