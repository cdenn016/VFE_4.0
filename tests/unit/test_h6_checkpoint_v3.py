from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable

import pytest
import torch
from torch import nn

from vfe4.training.checkpoint_v3 import (
    H6CheckpointV3,
    capture_h6_checkpoint_v3,
    decode_h6_checkpoint_v3,
    hydrate_h6_checkpoint_v3,
)
from vfe4.types.h6 import TrainingPhase
from vfe4.types.h6_prediction_v3 import (
    H6AttemptCursorV3,
    H6AttemptSpecV3,
    H6ObjectiveManifestV3,
    H6PredictionRuntimeIdentity,
    H6_DETERMINISTIC_POLICY_SHA256,
    H6_NO_COUNTER_CONSUMPTION_SHA256,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Failed to find (cuobjdump|nvdisasm)\\.exe:UserWarning"
)


def _sha(character: str) -> str:
    return character * 64


class _TinyState(nn.Module):
    def __init__(self, *, offset: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[1.0 + offset, -2.0], [0.5, 3.0 + offset]],
                dtype=torch.float64,
            )
        )
        self.bias = nn.Parameter(
            torch.tensor([0.25 + offset, -0.75], dtype=torch.float64)
        )
        self.register_buffer(
            "token_count",
            torch.tensor([[3, 5], [7, 11]], dtype=torch.int64),
        )


def _runtime(
    *,
    torch_full_version: str = "2.10.0.dev20251210+cu128",
) -> H6PredictionRuntimeIdentity:
    return H6PredictionRuntimeIdentity.create(
        python_version="3.13.5",
        torch_full_version=torch_full_version,
        cuda_runtime_version="12.8",
        cuda_device_name="NVIDIA GeForce RTX 5090",
        cuda_compute_capability=(12, 0),
    )


def _attempt(
    runtime: H6PredictionRuntimeIdentity,
    *,
    objective_kind: str = "complete_elbo",
    recognition_factory_sha256: str | None = _sha("6"),
) -> H6AttemptSpecV3:
    return H6AttemptSpecV3.create(
        git_head="a" * 40,
        dirty_digest=_sha("1"),
        readiness_sha256=_sha("2"),
        experiment_config_sha256=_sha("3"),
        endpoint_id="complete-a5",
        arm_id="A5",
        endpoint_config_sha256=_sha("4"),
        objective_kind=objective_kind,
        model_factory_sha256=_sha("5"),
        recognition_factory_sha256=recognition_factory_sha256,
        initialization_sha256=_sha("7"),
        optimizer_policy_sha256=_sha("8"),
        training_seed=23,
        data_identity_sha256=_sha("9"),
        window_schedule_sha256=_sha("a"),
        batch_schedule_sha256=_sha("b"),
        phase_schedule_sha256=_sha("c"),
        training_schedule_sha256=_sha("d"),
        recognition_estimator_sha256=_sha("e"),
        runtime_identity_sha256=runtime.runtime_identity_sha256,
    )


def _cursor(attempt: H6AttemptSpecV3) -> H6AttemptCursorV3:
    return H6AttemptCursorV3.create(
        attempt_spec_sha256=attempt.attempt_spec_sha256,
        pass_index=1,
        batch_index=17,
        next_phase=TrainingPhase.MODEL_ADAMW,
        example_ordinal=136,
        draw_block=11,
        counter_consumption_sha256=_sha("f"),
        permutation_sha256=_sha("0"),
        recognition_update_count=18,
        model_update_count=17,
        validation_boundary_count=2,
        checkpoint_boundary_count=3,
    )


def _model_ce_cursor(
    attempt: H6AttemptSpecV3,
    *,
    draw_block: int = 0,
    counter_consumption_sha256: str = H6_NO_COUNTER_CONSUMPTION_SHA256,
) -> H6AttemptCursorV3:
    return H6AttemptCursorV3.create(
        attempt_spec_sha256=attempt.attempt_spec_sha256,
        pass_index=0,
        batch_index=1,
        next_phase=TrainingPhase.MODEL_CE_ADAMW,
        example_ordinal=8,
        draw_block=draw_block,
        counter_consumption_sha256=counter_consumption_sha256,
        permutation_sha256=_sha("0"),
        recognition_update_count=0,
        model_update_count=1,
        checkpoint_boundary_count=1,
    )


def _objective(
    attempt: H6AttemptSpecV3,
    *,
    counter_consumption_sha256: str = _sha("f"),
    phase: TrainingPhase = TrainingPhase.RECOGNITION_ADAMW,
) -> H6ObjectiveManifestV3:
    return H6ObjectiveManifestV3.create(
        attempt_spec_sha256=attempt.attempt_spec_sha256,
        endpoint_config_sha256=attempt.endpoint_config_sha256,
        objective_kind=attempt.objective_kind,
        phase=phase,
        recognition_estimator_sha256=attempt.recognition_estimator_sha256,
        counter_consumption_sha256=counter_consumption_sha256,
        recognition_law_sha256=_sha("1"),
        detached_snapshot_sha256=_sha("2"),
        ordered_factor_bindings=(("emission", 0, _sha("3")),),
        total_raw_bytes_sha256=_sha("4"),
    )


def _adamw(
    groups: list[dict[str, object]],
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        groups,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _step(optimizer: torch.optim.AdamW) -> None:
    optimizer.zero_grad(set_to_none=True)
    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    sum(parameter.square().sum() for parameter in parameters).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _fixture() -> tuple[
    H6PredictionRuntimeIdentity,
    H6AttemptSpecV3,
    H6AttemptCursorV3,
    H6ObjectiveManifestV3,
    tuple[tuple[str, nn.Module], ...],
    tuple[tuple[str, torch.optim.AdamW], ...],
]:
    runtime = _runtime()
    attempt = _attempt(runtime)
    cursor = _cursor(attempt)
    objective = _objective(attempt)
    model = _TinyState(offset=0.0)
    recognition = _TinyState(offset=4.0)
    model_optimizer = _adamw(
        [
            {
                "params": [model.bias],
                "lr": 0.02,
                "weight_decay": 0.1,
            },
            {
                "params": [model.weight],
                "lr": 0.01,
                "weight_decay": 0.0,
            },
        ]
    )
    recognition_optimizer = _adamw(
        [
            {
                "params": [recognition.weight, recognition.bias],
                "lr": 0.03,
                "weight_decay": 0.2,
            }
        ]
    )
    _step(model_optimizer)
    _step(recognition_optimizer)
    return (
        runtime,
        attempt,
        cursor,
        objective,
        (("recognition", recognition), ("model", model)),
        (
            ("recognition", recognition_optimizer),
            ("model", model_optimizer),
        ),
    )


def _capture_fixture() -> tuple[
    H6CheckpointV3,
    tuple[tuple[str, nn.Module], ...],
    tuple[tuple[str, torch.optim.AdamW], ...],
]:
    runtime, attempt, cursor, objective, modules, optimizers = _fixture()
    checkpoint = capture_h6_checkpoint_v3(
        attempt_spec=attempt,
        cursor=cursor,
        objective_manifest=objective,
        runtime_identity=runtime,
        named_modules=modules,
        named_optimizers=optimizers,
    )
    return checkpoint, modules, optimizers


def _factory(
    offset: float,
    calls: list[H6AttemptSpecV3] | None = None,
) -> Callable[[H6AttemptSpecV3], nn.Module]:
    def construct(attempt: H6AttemptSpecV3) -> nn.Module:
        if calls is not None:
            calls.append(attempt)
        return _TinyState(offset=offset)

    return construct


def test_checkpoint_v3_canonicalizes_named_module_and_optimizer_state() -> None:
    runtime, attempt, cursor, objective, modules, optimizers = _fixture()
    first = capture_h6_checkpoint_v3(
        attempt_spec=attempt,
        cursor=cursor,
        objective_manifest=objective,
        runtime_identity=runtime,
        named_modules=modules,
        named_optimizers=optimizers,
    )
    reordered = capture_h6_checkpoint_v3(
        attempt_spec=attempt,
        cursor=cursor,
        objective_manifest=objective,
        runtime_identity=runtime,
        named_modules=tuple(reversed(modules)),
        named_optimizers=tuple(reversed(optimizers)),
    )

    encoded = first.to_bytes()
    assert encoded == reordered.to_bytes()
    assert first.checkpoint_sha256 == reordered.checkpoint_sha256
    decoded = decode_h6_checkpoint_v3(encoded)
    assert decoded.to_bytes() == encoded
    assert decoded.checkpoint_sha256 == first.checkpoint_sha256
    corrupted = bytearray(encoded)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="integrity"):
        decode_h6_checkpoint_v3(bytes(corrupted))
    module_names = tuple(record.name for record in first.module_tensors)
    assert module_names == tuple(sorted(module_names))
    assert module_names == (
        "model.bias",
        "model.token_count",
        "model.weight",
        "recognition.bias",
        "recognition.token_count",
        "recognition.weight",
    )
    for record in first.module_tensors:
        raw = record.raw_bytes()
        assert record.role in {"module_buffer", "module_parameter"}
        assert record.byte_order == "little"
        assert record.layout == "contiguous-row-major"
        assert record.byte_length == len(raw)
        assert record.raw_bytes_sha256 == hashlib.sha256(raw).hexdigest()
        assert isinstance(record.shape, tuple)
    token_record = next(
        record for record in first.module_tensors if record.name == "model.token_count"
    )
    assert token_record.dtype == "int64"
    assert token_record.shape == (2, 2)
    assert token_record.raw_bytes() == struct.pack("<qqqq", 3, 5, 7, 11)

    assert tuple(record.name for record in first.optimizers) == (
        "model",
        "recognition",
    )
    model_record = first.optimizers[0]
    assert tuple(group.name for group in model_record.groups) == (
        "model.group.000000",
        "model.group.000001",
    )
    assert model_record.groups[0].parameter_names == ("model.bias",)
    assert model_record.groups[1].parameter_names == ("model.weight",)
    assert tuple(state.parameter_name for state in model_record.states) == (
        "model.bias",
        "model.weight",
    )
    assert {
        tensor.state_name for state in model_record.states for tensor in state.tensors
    } == {"step", "exp_avg", "exp_avg_sq"}


def test_checkpoint_v3_rejects_duplicate_or_aliased_tensor_names() -> None:
    runtime = _runtime()
    attempt = _attempt(runtime)
    cursor = _cursor(attempt)
    objective = _objective(attempt)

    with pytest.raises(ValueError, match="duplicate"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=objective,
            runtime_identity=runtime,
            named_modules=(("model", _TinyState()), ("model", _TinyState())),
            named_optimizers=(),
        )
    with pytest.raises(ValueError, match="case-colliding"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=objective,
            runtime_identity=runtime,
            named_modules=(("model", _TinyState()), ("Model", _TinyState())),
            named_optimizers=(),
        )

    class Aliased(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = nn.Parameter(torch.ones(2, dtype=torch.float64))
            self.left = shared
            self.right = shared

    with pytest.raises(ValueError, match="shared-storage alias"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=objective,
            runtime_identity=runtime,
            named_modules=(("model", Aliased()),),
            named_optimizers=(),
        )

    class CaseCollision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lower = nn.Parameter(torch.ones(1, dtype=torch.float64))
            self.LOWER = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    with pytest.raises(ValueError, match="case-colliding"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=objective,
            runtime_identity=runtime,
            named_modules=(("model", CaseCollision()),),
            named_optimizers=(),
        )

    class BadState(nn.Module):
        def __init__(self, value: torch.Tensor) -> None:
            super().__init__()
            self.parameter = nn.Parameter(torch.ones(1, dtype=torch.float64))
            self.register_buffer("bad", value)

    bad_values = (
        torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1.0], dtype=torch.float64),
            (1,),
        ),
        torch.ones(1, dtype=torch.complex128),
        torch.ones(1, dtype=torch.float32),
    )
    for value in bad_values:
        with pytest.raises(ValueError, match="sparse|unsupported|float64"):
            capture_h6_checkpoint_v3(
                attempt_spec=attempt,
                cursor=cursor,
                objective_manifest=objective,
                runtime_identity=runtime,
                named_modules=(("model", BadState(value)),),
                named_optimizers=(),
            )

    _, _, _, _, modules, optimizers = _fixture()
    first_parameter = optimizers[0][1].param_groups[0]["params"][0]
    optimizers[0][1].state[first_parameter]["unknown"] = torch.zeros(
        1, dtype=torch.float64
    )
    with pytest.raises(ValueError, match="unknown AdamW state"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=objective,
            runtime_identity=runtime,
            named_modules=modules,
            named_optimizers=optimizers,
        )
    optimizers[0][1].state[first_parameter].pop("unknown")

    ghost = nn.Parameter(torch.ones(1, dtype=torch.float64))
    optimizers[0][1].state[ghost] = {}
    with pytest.raises(ValueError, match="unbound optimizer state"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=objective,
            runtime_identity=runtime,
            named_modules=modules,
            named_optimizers=optimizers,
        )


def test_fresh_cpu_hydration_restores_named_optimizer_groups() -> None:
    checkpoint, source_modules, source_optimizers = _capture_fixture()
    calls: list[H6AttemptSpecV3] = []
    hydrated = hydrate_h6_checkpoint_v3(
        checkpoint,
        expected_attempt_spec=checkpoint.attempt_spec,
        expected_runtime_identity=checkpoint.runtime_identity,
        live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
        module_factories=(
            ("recognition", _factory(-20.0, calls)),
            ("model", _factory(20.0, calls)),
        ),
        authorized_device="cpu",
        allow_synthetic_cpu=True,
    )
    assert calls == [checkpoint.attempt_spec, checkpoint.attempt_spec]

    source_by_name = dict(source_modules)
    hydrated_by_name = dict(hydrated.named_modules)
    assert set(hydrated_by_name) == {"model", "recognition"}
    for module_name, module in hydrated_by_name.items():
        assert module is not source_by_name[module_name]
        assert tuple(module.state_dict()) == tuple(
            source_by_name[module_name].state_dict()
        )
        for name, value in module.state_dict().items():
            assert torch.equal(
                value,
                source_by_name[module_name].state_dict()[name],
            )
            assert value.device.type == "cpu"

    parameter_names = {
        id(parameter): f"{module_name}.{local_name}"
        for module_name, module in hydrated.named_modules
        for local_name, parameter in module.named_parameters()
    }
    hydrated_optimizers = dict(hydrated.named_optimizers)
    model_groups = hydrated_optimizers["model"].param_groups
    assert tuple(
        tuple(parameter_names[id(parameter)] for parameter in group["params"])
        for group in model_groups
    ) == (("model.bias",), ("model.weight",))
    assert tuple(group["lr"] for group in model_groups) == (0.02, 0.01)
    assert tuple(group["weight_decay"] for group in model_groups) == (0.1, 0.0)

    source_optimizer_by_name = dict(source_optimizers)
    for optimizer_name, optimizer in hydrated.named_optimizers:
        source_optimizer = source_optimizer_by_name[optimizer_name]
        source_parameters = {
            f"{optimizer_name}.{local_name}": parameter
            for local_name, parameter in source_by_name[
                optimizer_name
            ].named_parameters()
        }
        hydrated_parameters = {
            f"{optimizer_name}.{local_name}": parameter
            for local_name, parameter in hydrated_by_name[
                optimizer_name
            ].named_parameters()
        }
        for parameter_name in hydrated_parameters:
            source_state = source_optimizer.state[source_parameters[parameter_name]]
            state = optimizer.state[hydrated_parameters[parameter_name]]
            assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
            for name, value in state.items():
                assert torch.equal(value, source_state[name])
                assert value.device.type == "cpu"

    class MissingBias(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones((2, 2), dtype=torch.float64))
            self.register_buffer(
                "token_count",
                torch.tensor([[3, 5], [7, 11]], dtype=torch.int64),
            )

    with pytest.raises(ValueError, match="module inventory"):
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=checkpoint.attempt_spec,
            expected_runtime_identity=checkpoint.runtime_identity,
            live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
            module_factories=(
                ("model", lambda _: MissingBias()),
                ("recognition", _factory(0.0)),
            ),
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )


def test_cursor_restores_next_phase_batch_and_counter_coordinates() -> None:
    runtime, attempt, cursor, _, modules, optimizers = _fixture()
    with pytest.raises(ValueError, match="counter consumption"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=_objective(
                attempt,
                counter_consumption_sha256=_sha("e"),
            ),
            runtime_identity=runtime,
            named_modules=modules,
            named_optimizers=optimizers,
        )
    with pytest.raises(ValueError, match="phase transition"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=_objective(
                attempt,
                phase=TrainingPhase.MODEL_CE_ADAMW,
            ),
            runtime_identity=runtime,
            named_modules=modules,
            named_optimizers=optimizers,
        )

    checkpoint, _, _ = _capture_fixture()
    hydrated = hydrate_h6_checkpoint_v3(
        checkpoint,
        expected_attempt_spec=checkpoint.attempt_spec,
        expected_runtime_identity=checkpoint.runtime_identity,
        live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
        module_factories=(
            ("model", _factory(0.0)),
            ("recognition", _factory(0.0)),
        ),
        authorized_device="cpu",
        allow_synthetic_cpu=True,
    )

    cursor = hydrated.cursor
    assert cursor is checkpoint.cursor
    assert (
        cursor.pass_index,
        cursor.batch_index,
        cursor.next_phase,
        cursor.example_ordinal,
        cursor.sample_ordinal,
        cursor.draw_block,
    ) == (1, 17, TrainingPhase.MODEL_ADAMW, 136, 0, 11)
    assert cursor.counter_consumption_sha256 == _sha("f")
    assert cursor.permutation_sha256 == _sha("0")
    assert (
        cursor.recognition_update_count,
        cursor.model_update_count,
        cursor.validation_boundary_count,
        cursor.checkpoint_boundary_count,
    ) == (18, 17, 2, 3)


def test_no_latent_cross_entropy_uses_model_ce_checkpoint_phase() -> None:
    runtime = _runtime()
    attempt = _attempt(
        runtime,
        objective_kind="cross_entropy",
        recognition_factory_sha256=None,
    )
    cursor = _model_ce_cursor(attempt)
    model = _TinyState()
    optimizer = _adamw(
        [
            {
                "params": [model.weight, model.bias],
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    _step(optimizer)

    checkpoint = capture_h6_checkpoint_v3(
        attempt_spec=attempt,
        cursor=cursor,
        objective_manifest=_objective(
            attempt,
            counter_consumption_sha256=H6_NO_COUNTER_CONSUMPTION_SHA256,
            phase=TrainingPhase.MODEL_CE_ADAMW,
        ),
        runtime_identity=runtime,
        named_modules=(("model", model),),
        named_optimizers=(("model", optimizer),),
    )
    decoded = decode_h6_checkpoint_v3(checkpoint.to_bytes())

    assert decoded.to_bytes() == checkpoint.to_bytes()
    assert decoded.cursor.next_phase is TrainingPhase.MODEL_CE_ADAMW
    assert decoded.attempt_spec.objective_kind == "cross_entropy"
    assert decoded.objective_manifest.is_elbo is False


def test_model_ce_checkpoint_rejects_complete_elbo() -> None:
    runtime = _runtime()
    attempt = _attempt(runtime, recognition_factory_sha256=None)
    cursor = _model_ce_cursor(attempt)
    model = _TinyState()
    optimizer = _adamw(
        [
            {
                "params": [model.weight, model.bias],
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    _step(optimizer)

    with pytest.raises(ValueError, match="objective/cursor phase transition"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=_objective(
                attempt,
                counter_consumption_sha256=H6_NO_COUNTER_CONSUMPTION_SHA256,
                phase=TrainingPhase.MODEL_CE_ADAMW,
            ),
            runtime_identity=runtime,
            named_modules=(("model", model),),
            named_optimizers=(("model", optimizer),),
        )


def test_model_ce_checkpoint_rejects_renamed_posterior_root() -> None:
    runtime = _runtime()
    attempt = _attempt(
        runtime,
        objective_kind="cross_entropy",
        recognition_factory_sha256=None,
    )
    cursor = _model_ce_cursor(attempt)
    model = _TinyState()
    posterior = _TinyState(offset=4.0)
    model_optimizer = _adamw(
        [
            {
                "params": [model.weight, model.bias],
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    posterior_optimizer = _adamw(
        [
            {
                "params": [posterior.weight, posterior.bias],
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    _step(model_optimizer)
    _step(posterior_optimizer)

    with pytest.raises(ValueError, match="root inventory"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=_objective(
                attempt,
                counter_consumption_sha256=H6_NO_COUNTER_CONSUMPTION_SHA256,
                phase=TrainingPhase.MODEL_CE_ADAMW,
            ),
            runtime_identity=runtime,
            named_modules=(("model", model), ("posterior", posterior)),
            named_optimizers=(
                ("model", model_optimizer),
                ("posterior", posterior_optimizer),
            ),
        )


@pytest.mark.parametrize(
    ("draw_block", "counter_consumption_sha256"),
    (
        (1, H6_NO_COUNTER_CONSUMPTION_SHA256),
        (0, _sha("f")),
    ),
)
def test_model_ce_checkpoint_rejects_counter_consumption(
    draw_block: int,
    counter_consumption_sha256: str,
) -> None:
    runtime = _runtime()
    attempt = _attempt(
        runtime,
        objective_kind="cross_entropy",
        recognition_factory_sha256=None,
    )
    cursor = _model_ce_cursor(
        attempt,
        draw_block=draw_block,
        counter_consumption_sha256=counter_consumption_sha256,
    )
    model = _TinyState()
    optimizer = _adamw(
        [
            {
                "params": [model.weight, model.bias],
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    _step(optimizer)

    with pytest.raises(ValueError, match="zero counter consumption"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=_objective(
                attempt,
                counter_consumption_sha256=counter_consumption_sha256,
                phase=TrainingPhase.MODEL_CE_ADAMW,
            ),
            runtime_identity=runtime,
            named_modules=(("model", model),),
            named_optimizers=(("model", optimizer),),
        )


def test_checkpoint_rejects_optimizer_bound_to_different_module_root() -> None:
    runtime, attempt, cursor, objective, modules, _ = _fixture()
    module_by_name = dict(modules)
    model = module_by_name["model"]
    recognition = module_by_name["recognition"]
    swapped_model_optimizer = _adamw(
        [
            {
                "params": list(recognition.parameters()),
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    swapped_recognition_optimizer = _adamw(
        [
            {
                "params": list(model.parameters()),
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    _step(swapped_model_optimizer)
    _step(swapped_recognition_optimizer)

    with pytest.raises(ValueError, match="same-root"):
        capture_h6_checkpoint_v3(
            attempt_spec=attempt,
            cursor=cursor,
            objective_manifest=objective,
            runtime_identity=runtime,
            named_modules=modules,
            named_optimizers=(
                ("model", swapped_model_optimizer),
                ("recognition", swapped_recognition_optimizer),
            ),
        )


def test_resume_rejects_runtime_or_determinism_drift() -> None:
    checkpoint, _, _ = _capture_fixture()
    factory_calls: list[H6AttemptSpecV3] = []
    factories = (
        ("model", _factory(0.0, factory_calls)),
        ("recognition", _factory(0.0, factory_calls)),
    )

    with pytest.raises(RuntimeError, match="runtime identity drift"):
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=checkpoint.attempt_spec,
            expected_runtime_identity=_runtime(
                torch_full_version="2.10.1+cu128",
            ),
            live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
            module_factories=factories,
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )
    with pytest.raises(RuntimeError, match="deterministic policy drift"):
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=checkpoint.attempt_spec,
            expected_runtime_identity=checkpoint.runtime_identity,
            live_deterministic_policy_sha256=_sha("d"),
            module_factories=factories,
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )
    assert factory_calls == []
