from __future__ import annotations

import inspect
import tracemalloc
from dataclasses import FrozenInstanceError

import pytest
import torch

import vfe4.inference as inference_package
import vfe4.inference.h4_instrumentation as instrumentation_module
from vfe4.inference.h4_instrumentation import (
    CountingOperationRecorder,
    H4OperationKind as InstrumentationOperationKind,
    InstrumentedLinearAlgebra,
    NullOperationRecorder,
    measure_untimed_memory,
)
from vfe4.types.h4 import (
    H4MemoryRecord,
    H4OperationKind as TypesOperationKind,
    H4OperationRecord,
)


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


def test_recorder_mutation_revalidates_private_capability_at_every_endpoint() -> None:
    record = H4OperationRecord(
        "problem-capability",
        "information",
        "cholesky",
        ((1, 1),),
        (1, 1),
        1,
    )
    for recorder in (NullOperationRecorder(), CountingOperationRecorder()):
        InstrumentedLinearAlgebra(
            problem_id="problem-capability",
            arm="information",
            recorder=recorder,
        )
        before = recorder.snapshot()
        with pytest.raises(TypeError):
            recorder._accept(record)  # type: ignore[call-arg]
        with pytest.raises(PermissionError, match="capability"):
            recorder._accept(object(), record)
        with pytest.raises(TypeError):
            recorder._accept_authorized(record)  # type: ignore[call-arg]
        with pytest.raises(PermissionError, match="capability"):
            recorder._accept_authorized(object(), record)
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


def test_facade_binding_and_spd_receipts_require_the_exact_unchanged_facade_call() -> None:
    class FacadeSubclass(InstrumentedLinearAlgebra):
        pass

    value = torch.tensor(((2.0, 0.25), (0.25, 1.5)), dtype=torch.float64)
    facade = InstrumentedLinearAlgebra(
        problem_id="problem-proof",
        arm="moment",
        recorder=NullOperationRecorder(),
    )
    factor = facade.cholesky(value)
    matrix, proof = instrumentation_module._facade_proven_spd_tuple(
        facade,
        value,
        factor,
    )
    assert matrix == ((2.0, 0.25), (0.25, 1.5))
    assert proof is not None
    assert facade._cholesky_receipt is None
    assert not hasattr(facade, "_cholesky_receipts")
    with pytest.raises(FrozenInstanceError):
        proof._source = "spoofed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="receipt"):
        instrumentation_module._facade_proven_spd_tuple(facade, value, factor)

    other_facade = InstrumentedLinearAlgebra(
        problem_id="problem-proof",
        arm="moment",
        recorder=NullOperationRecorder(),
    )
    with pytest.raises(ValueError, match="receipt"):
        instrumentation_module._facade_proven_spd_tuple(
            other_facade,
            value,
            factor,
        )

    mismatched_value = value.clone()
    mismatched_factor = facade.cholesky(mismatched_value)
    with pytest.raises(ValueError, match="receipt"):
        instrumentation_module._facade_proven_spd_tuple(
            facade,
            mismatched_value.clone(),
            mismatched_factor,
        )
    with pytest.raises(ValueError, match="receipt"):
        instrumentation_module._facade_proven_spd_tuple(
            facade,
            mismatched_value,
            mismatched_factor,
        )

    changed_value = value.clone()
    changed_factor = facade.cholesky(changed_value)
    changed_value.add_(torch.eye(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="changed"):
        instrumentation_module._facade_proven_spd_tuple(
            facade,
            changed_value,
            changed_factor,
        )

    storage_value = torch.tensor(((3.0, 0.5), (0.5, 2.0)), dtype=torch.float64)
    storage_factor = facade.cholesky(storage_value)
    storage_value.data = storage_value.clone().data
    with pytest.raises(ValueError, match="changed"):
        instrumentation_module._facade_proven_spd_tuple(
            facade,
            storage_value,
            storage_factor,
        )

    stale_value = value.clone()
    stale_factor = facade.cholesky(stale_value)
    with pytest.raises(RuntimeError):
        facade.cholesky(
            torch.tensor(((1.0, 2.0), (2.0, 1.0)), dtype=torch.float64)
        )
    assert facade._cholesky_receipt is None
    with pytest.raises(ValueError, match="receipt"):
        instrumentation_module._facade_proven_spd_tuple(
            facade,
            stale_value,
            stale_factor,
        )

    subclass = FacadeSubclass(
        problem_id="problem-proof",
        arm="moment",
        recorder=NullOperationRecorder(),
    )
    with pytest.raises(ValueError, match="exact"):
        instrumentation_module._facade_binding(subclass)


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


def test_untimed_memory_preserves_ambient_tracemalloc_state_and_peak() -> None:
    if tracemalloc.is_tracing():
        pytest.skip("test requires ownership of an initially stopped tracemalloc state")

    tracemalloc.start()
    try:
        peak_seed = bytearray(1_000_000)
        del peak_seed
        before_peak = tracemalloc.get_traced_memory()[1]

        record = measure_untimed_memory("problem-ambient", "moment", lambda: None)
        assert tracemalloc.is_tracing()
        assert tracemalloc.get_traced_memory()[1] == before_peak
        assert record.python_peak_bytes is None
        assert record.unavailable_fields == (
            "python_peak_bytes",
            "process_working_set_delta_bytes",
        )

        def fail() -> object:
            raise RuntimeError("ambient-boom")

        with pytest.raises(RuntimeError, match="ambient-boom"):
            measure_untimed_memory("problem-ambient", "moment", fail)
        assert tracemalloc.is_tracing()
        assert tracemalloc.get_traced_memory()[1] == before_peak
    finally:
        tracemalloc.stop()

    assert not tracemalloc.is_tracing()
    with pytest.raises(RuntimeError, match="cold-boom"):
        measure_untimed_memory(
            "problem-cold",
            "information",
            lambda: (_ for _ in ()).throw(RuntimeError("cold-boom")),
        )
    assert not tracemalloc.is_tracing()


def test_inference_package_reexports_the_existing_operation_kind_alias() -> None:
    assert inference_package.H4OperationKind is InstrumentationOperationKind
    assert inference_package.H4OperationKind is TypesOperationKind
