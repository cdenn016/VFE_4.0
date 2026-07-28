from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from vfe4.training import checkpoint_v3 as checkpoint_v3_module
from vfe4.training.checkpoint_v3 import (
    H6CheckpointV3,
    capture_h6_checkpoint_v3,
    decode_h6_checkpoint_v3,
    hydrate_h6_checkpoint_v3,
    read_h6_checkpoint_file_v3,
)
from vfe4.training.h6_engine_v3 import (
    H6BatchLiveRecognitionStateV3,
    H6DetachedBatchRecognitionSnapshotV3,
    H6EngineAuthorityV3,
    H6LiveRecognitionStateV3,
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


class _OriginalSemanticState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[1.0, -2.0], [0.5, 3.0]],
                dtype=torch.float64,
            )
        )
        self.bias = nn.Parameter(torch.tensor([0.25, -0.75], dtype=torch.float64))
        self.register_buffer(
            "token_count",
            torch.tensor([[3, 5], [7, 11]], dtype=torch.int64),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.project(inputs) + self.bias

    def project(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight.T


class _AlternateSemanticState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[1.0, -2.0], [0.5, 3.0]],
                dtype=torch.float64,
            )
        )
        self.bias = nn.Parameter(torch.tensor([0.25, -0.75], dtype=torch.float64))
        self.register_buffer(
            "token_count",
            torch.tensor([[3, 5], [7, 11]], dtype=torch.int64),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs @ self.weight + self.bias


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


def _fixture_factory_authority(
    checkpoint: H6CheckpointV3,
    *,
    module_factories: object,
    expected_named_modules: object | None = None,
) -> object:
    if expected_named_modules is None:
        expected_named_modules = (
            ("model", _TinyState()),
            ("recognition", _TinyState()),
        )
    return checkpoint_v3_module._issue_h6_checkpoint_factory_authority_v3(
        attempt_spec=checkpoint.attempt_spec,
        expected_named_modules=expected_named_modules,
        module_factories=module_factories,
    )


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
    for decoded_optimizer, captured_optimizer in zip(
        decoded.optimizers,
        first.optimizers,
        strict=True,
    ):
        assert decoded_optimizer.groups == captured_optimizer.groups
        for group in decoded_optimizer.groups:
            hyperparameters = dict(group.hyperparameters)
            assert type(hyperparameters["lr"]) is float
            assert type(hyperparameters["weight_decay"]) is float
            assert type(hyperparameters["eps"]) is float
            assert type(hyperparameters["betas"]) is tuple and all(
                type(value) is float for value in hyperparameters["betas"]
            )
        for state in decoded_optimizer.states:
            assert all(
                type(value) in (int, float) and not isinstance(value, bool)
                for _name, value in state.scalars
            )
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


def test_checkpoint_file_reader_is_bounded_and_digest_bound(
    tmp_path: Path,
) -> None:
    checkpoint, _, _ = _capture_fixture()
    encoded = checkpoint.to_bytes()
    path = tmp_path / "checkpoint.h6v3"
    path.write_bytes(encoded)

    reopened = read_h6_checkpoint_file_v3(
        path,
        maximum_bytes=len(encoded),
        expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
    )
    assert reopened.to_bytes() == encoded
    assert reopened.checkpoint_sha256 == checkpoint.checkpoint_sha256

    with pytest.raises(ValueError, match="bounded|maximum"):
        read_h6_checkpoint_file_v3(
            path,
            maximum_bytes=len(encoded) - 1,
            expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
        )
    with pytest.raises(ValueError, match="expected|digest"):
        read_h6_checkpoint_file_v3(
            path,
            maximum_bytes=len(encoded),
            expected_checkpoint_sha256=_sha("0"),
        )

    corrupted = bytearray(encoded)
    corrupted[-1] ^= 1
    corrupt_path = tmp_path / "corrupt.h6v3"
    corrupt_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="integrity|digest"):
        read_h6_checkpoint_file_v3(
            corrupt_path,
            maximum_bytes=len(corrupted),
            expected_checkpoint_sha256=checkpoint.checkpoint_sha256,
        )


def test_checkpoint_v3_roundtrips_exact_mid_model_batch_snapshot() -> None:
    runtime, attempt, cursor, objective, modules, optimizers = _fixture()
    authority = H6EngineAuthorityV3.create(
        attempt_spec_sha256=attempt.attempt_spec_sha256,
        endpoint_config_sha256=attempt.endpoint_config_sha256,
        readiness_sha256=_sha("1"),
        readiness_matching_set_sha256=_sha("2"),
        matching_set_sha256=_sha("2"),
        matching_policy_sha256=_sha("3"),
        readiness_training_schedule_sha256=_sha("4"),
        training_schedule_sha256=_sha("4"),
        readiness_runtime_identity_sha256=runtime.runtime_identity_sha256,
        runtime_identity_sha256=runtime.runtime_identity_sha256,
        planned_attempt_sha256=_sha("5"),
        endpoint_config_id=attempt.endpoint_id,
        matching_ledger_sha256=_sha("6"),
        matching_report_sha256s=(_sha("7"),),
        receiver_count=2,
        state_categorical_enabled=False,
        model_categorical_enabled=False,
        tuning_cell_sha256=_sha("8"),
        optimizer_policy_sha256=(
            "67b498399b293d4f267cb7ffbe5f0e329ac0025adaaa5f86869588ad720f5ce8"
        ),
        optimizer_learning_rate=1.0e-3,
        optimizer_weight_decay=0.0,
        objective_kind="complete_elbo",
        latent_enabled=True,
    )
    state = H6LiveRecognitionStateV3.create(
        endpoint_config_sha256=attempt.endpoint_config_sha256,
        receiver_count=2,
        state_categorical_enabled=False,
        model_categorical_enabled=False,
        state_categorical_supports=(None,),
        model_categorical_supports=(None,),
        receiver_components=((0, ("ordinary",)), (1, ("terminal",))),
        tensors={
            "receiver.0.component.ordinary.mean": torch.ones(
                2, dtype=torch.float64, requires_grad=True
            ),
            "receiver.0.shared_precision_cholesky": torch.eye(
                2, dtype=torch.float64, requires_grad=True
            ),
            "receiver.1.component.terminal.mean": torch.full(
                (2,), 2.0, dtype=torch.float64, requires_grad=True
            ),
            "receiver.1.shared_precision_cholesky": torch.eye(
                2, dtype=torch.float64, requires_grad=True
            ),
            "state.absent.support": torch.tensor((-1,), dtype=torch.int64),
            "state.absent.categorical_row": torch.ones(1, dtype=torch.float64),
            "model.absent.support": torch.tensor((-1,), dtype=torch.int64),
            "model.absent.categorical_row": torch.ones(1, dtype=torch.float64),
        },
        context_sha256=_sha("9"),
        recognition_state_sha256=_sha("a"),
        source_model_sha256=_sha("b"),
        law_sha256=_sha("c"),
    )
    live_batch = H6BatchLiveRecognitionStateV3.create(
        authority=authority,
        states=(state,),
        active_target_counts=(1,),
        active_receiver_masks=((True, True),),
    )
    snapshot = H6DetachedBatchRecognitionSnapshotV3.capture(
        live_batch,
        authority=authority,
        post_recognition_cursor=cursor,
    )

    checkpoint = capture_h6_checkpoint_v3(
        attempt_spec=attempt,
        cursor=cursor,
        objective_manifest=objective,
        runtime_identity=runtime,
        named_modules=modules,
        named_optimizers=optimizers,
        detached_batch_snapshot=snapshot,
    )
    reopened = decode_h6_checkpoint_v3(checkpoint.to_bytes())

    assert type(reopened.detached_batch_snapshot) is (
        H6DetachedBatchRecognitionSnapshotV3
    )
    assert reopened.detached_batch_snapshot.snapshot_sha256 == snapshot.snapshot_sha256
    assert reopened.detached_batch_snapshot.names == snapshot.names
    for name in snapshot.names:
        torch.testing.assert_close(
            reopened.detached_batch_snapshot[name],
            snapshot[name],
        )


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


@pytest.mark.parametrize(
    "unauthorized_kind",
    ("alternate-class", "instance-forward-override"),
)
def test_hydration_rejects_unauthorized_forward_semantics_before_state_load(
    monkeypatch: pytest.MonkeyPatch,
    unauthorized_kind: str,
) -> None:
    runtime = _runtime()
    attempt = _attempt(
        runtime,
        objective_kind="cross_entropy",
        recognition_factory_sha256=None,
    )
    original = _OriginalSemanticState()
    original_optimizer = _adamw(
        [
            {
                "params": [original.weight, original.bias],
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    _step(original_optimizer)
    checkpoint = capture_h6_checkpoint_v3(
        attempt_spec=attempt,
        cursor=_model_ce_cursor(attempt),
        objective_manifest=_objective(
            attempt,
            counter_consumption_sha256=H6_NO_COUNTER_CONSUMPTION_SHA256,
            phase=TrainingPhase.MODEL_CE_ADAMW,
        ),
        runtime_identity=runtime,
        named_modules=(("model", original),),
        named_optimizers=(("model", original_optimizer),),
    )

    unauthorized: nn.Module
    if unauthorized_kind == "alternate-class":
        unauthorized = _AlternateSemanticState()
    else:
        unauthorized = _OriginalSemanticState()
        unauthorized.forward = lambda inputs: inputs - 1.0  # type: ignore[method-assign]
    expected = _OriginalSemanticState()
    assert tuple(expected.state_dict()) == tuple(unauthorized.state_dict())
    assert tuple(
        (name, parameter.dtype, tuple(parameter.shape))
        for name, parameter in expected.named_parameters()
    ) == tuple(
        (name, parameter.dtype, tuple(parameter.shape))
        for name, parameter in unauthorized.named_parameters()
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in unauthorized.state_dict().items()
    }
    factory_calls: list[H6AttemptSpecV3] = []

    def unauthorized_factory(
        bound_attempt: H6AttemptSpecV3,
    ) -> nn.Module:
        factory_calls.append(bound_attempt)
        return unauthorized

    authority = checkpoint_v3_module._issue_h6_checkpoint_factory_authority_v3(
        attempt_spec=attempt,
        expected_named_modules=(("model", expected),),
        module_factories=(("model", unauthorized_factory),),
    )
    load_calls: list[bool] = []

    def forbidden_load_module_state(**_kwargs: object) -> None:
        load_calls.append(True)
        raise AssertionError("unauthorized module reached checkpoint state load")

    monkeypatch.setattr(
        checkpoint_v3_module,
        "_load_module_state",
        forbidden_load_module_state,
    )
    with pytest.raises(ValueError, match="semantic/type signature|forward override"):
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=attempt,
            expected_runtime_identity=runtime,
            live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
            factory_authority=authority,
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )

    assert factory_calls == [attempt]
    assert load_calls == []
    for name, value in unauthorized.state_dict().items():
        assert torch.equal(value, before[name])


def test_checkpoint_factory_authority_has_no_public_issuer_surface() -> None:
    assert (
        "bind_h6_checkpoint_factory_authority_v3"
        not in checkpoint_v3_module.__all__
    )
    assert "H6CheckpointFactoryAuthorityV3" not in checkpoint_v3_module.__all__
    assert not hasattr(
        checkpoint_v3_module,
        "bind_h6_checkpoint_factory_authority_v3",
    )
    assert not hasattr(
        checkpoint_v3_module,
        "H6CheckpointFactoryAuthorityV3",
    )


def test_replaced_checkpoint_factory_authority_is_rejected_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _, _ = _capture_fixture()
    authority = _fixture_factory_authority(
        checkpoint,
        module_factories=(
            ("model", _factory(0.0)),
            ("recognition", _factory(0.0)),
        ),
    )
    copied_authority = replace(authority)
    load_calls: list[bool] = []

    def forbidden_load_module_state(**_kwargs: object) -> None:
        load_calls.append(True)
        raise AssertionError("copied authority reached checkpoint state load")

    monkeypatch.setattr(
        checkpoint_v3_module,
        "_load_module_state",
        forbidden_load_module_state,
    )
    with pytest.raises(ValueError, match="sealed|issued|consumed"):
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=checkpoint.attempt_spec,
            expected_runtime_identity=checkpoint.runtime_identity,
            live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
            factory_authority=copied_authority,  # type: ignore[arg-type]
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )

    assert load_calls == []


def test_nonforward_behavior_mutation_after_issuance_is_rejected_before_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    attempt = _attempt(
        runtime,
        objective_kind="cross_entropy",
        recognition_factory_sha256=None,
    )
    original = _OriginalSemanticState()
    optimizer = _adamw(
        [
            {
                "params": [original.weight, original.bias],
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        ]
    )
    _step(optimizer)
    checkpoint = capture_h6_checkpoint_v3(
        attempt_spec=attempt,
        cursor=_model_ce_cursor(attempt),
        objective_manifest=_objective(
            attempt,
            counter_consumption_sha256=H6_NO_COUNTER_CONSUMPTION_SHA256,
            phase=TrainingPhase.MODEL_CE_ADAMW,
        ),
        runtime_identity=runtime,
        named_modules=(("model", original),),
        named_optimizers=(("model", optimizer),),
    )
    authority = checkpoint_v3_module._issue_h6_checkpoint_factory_authority_v3(
        attempt_spec=attempt,
        expected_named_modules=(("model", _OriginalSemanticState()),),
        module_factories=(("model", lambda _attempt: _OriginalSemanticState()),),
    )

    def altered_project(
        self: _OriginalSemanticState,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return inputs @ self.weight

    monkeypatch.setattr(_OriginalSemanticState, "project", altered_project)
    load_calls: list[bool] = []

    def forbidden_load_module_state(**_kwargs: object) -> None:
        load_calls.append(True)
        raise AssertionError("mutated behavior reached checkpoint state load")

    monkeypatch.setattr(
        checkpoint_v3_module,
        "_load_module_state",
        forbidden_load_module_state,
    )
    with pytest.raises(ValueError, match="semantic|behavior"):
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=attempt,
            expected_runtime_identity=runtime,
            live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
            factory_authority=authority,  # type: ignore[arg-type]
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )

    assert load_calls == []


def test_fresh_cpu_hydration_restores_named_optimizer_groups() -> None:
    checkpoint, source_modules, source_optimizers = _capture_fixture()
    calls: list[H6AttemptSpecV3] = []
    factory_authority = _fixture_factory_authority(
        checkpoint,
        module_factories=(
            ("recognition", _factory(-20.0, calls)),
            ("model", _factory(20.0, calls)),
        ),
    )
    hydrated = hydrate_h6_checkpoint_v3(
        checkpoint,
        expected_attempt_spec=checkpoint.attempt_spec,
        expected_runtime_identity=checkpoint.runtime_identity,
        live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
        factory_authority=factory_authority,
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
        factory_authority = _fixture_factory_authority(
            checkpoint,
            expected_named_modules=(
                ("model", MissingBias()),
                ("recognition", _TinyState()),
            ),
            module_factories=(
                ("model", lambda _: MissingBias()),
                ("recognition", _factory(0.0)),
            ),
        )
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=checkpoint.attempt_spec,
            expected_runtime_identity=checkpoint.runtime_identity,
            live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
            factory_authority=factory_authority,
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
    factory_authority = _fixture_factory_authority(
        checkpoint,
        module_factories=(
            ("model", _factory(0.0)),
            ("recognition", _factory(0.0)),
        ),
    )
    hydrated = hydrate_h6_checkpoint_v3(
        checkpoint,
        expected_attempt_spec=checkpoint.attempt_spec,
        expected_runtime_identity=checkpoint.runtime_identity,
        live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
        factory_authority=factory_authority,
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
    factory_authority = _fixture_factory_authority(
        checkpoint,
        module_factories=factories,
    )

    with pytest.raises(RuntimeError, match="runtime identity drift"):
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=checkpoint.attempt_spec,
            expected_runtime_identity=_runtime(
                torch_full_version="2.10.1+cu128",
            ),
            live_deterministic_policy_sha256=H6_DETERMINISTIC_POLICY_SHA256,
            factory_authority=factory_authority,
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )
    with pytest.raises(RuntimeError, match="deterministic policy drift"):
        hydrate_h6_checkpoint_v3(
            checkpoint,
            expected_attempt_spec=checkpoint.attempt_spec,
            expected_runtime_identity=checkpoint.runtime_identity,
            live_deterministic_policy_sha256=_sha("d"),
            factory_authority=factory_authority,
            authorized_device="cpu",
            allow_synthetic_cpu=True,
        )
    assert factory_calls == []
