from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

import vfe4.training.h6_training_attempt_v3 as attempt_v3
from vfe4.artifacts.h6_prediction_v3 import H6PredictionV3Authorities
from vfe4.config import (
    H6PredictionV3ResolvedConfig,
    resolve_h6_prediction_v3_config,
)
from vfe4.data.access import H6TrainingDataV3
from vfe4.data.byte_tokenizer import ByteTokenizerV1
from vfe4.data.windows import CausalPrefix, build_causal_windows
from vfe4.evaluation.smc_uncertainty import SMC_BIAS_SEMANTICS
from vfe4.numerics.critical_values import CRITICAL_VALUES_PROTOCOL_SHA256
from vfe4.training.h6_execution_v3 import (
    H6ExecutableAttemptV3,
    bind_h6_executable_attempt_v3,
)
from vfe4.training.h6_experiment_v3 import (
    H6ExperimentPlanV3,
    H6PlannedAttemptV3,
    plan_h6_experiment_v3,
)
from vfe4.training.h6_matching_v3 import (
    H6_MATCHING_POLICY_V3,
    H6MatchingSetV3,
    build_h6_matching_set_v3,
)
from vfe4.training.h6_readiness import (
    _derive_h6_prediction_readiness_v3,
)
from vfe4.training.h6_runtime_v3 import (
    H6InstalledRuntimeBindingV3,
    configure_installed_runtime_v3,
)
from vfe4.training.matching import (
    ARM_MATRIX_ROWS,
    H6_ADAMW_POLICY,
    arm_matrix_sha256,
)
from vfe4.types import H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
from vfe4.types.h6 import (
    EndpointSmcProtocol,
    H6ArmPhaseSchedule,
    H6OuterSchedule,
    ObjectiveGateSpec,
    TrainingPhase,
)
from vfe4.types.h6_prediction_v3 import (
    H6_CHECKPOINT_CODEC_SHA256,
    H6_COUNTER_MAPPING_SHA256,
    H6_PHASE_OWNERSHIP_SHA256,
    H6_SCORING_INVENTORY_SHA256,
    H6PredictionRuntimeIdentity,
    H6PredictionV3ReadinessToken,
    H6RecognitionEstimatorSpec,
    H6TrainingScheduleV3,
)


_GIT_HEAD = "4" * 40
_DIRTY_DIGEST = "5" * 64
_DATA_IDENTITY_SHA256 = hashlib.sha256(b"cuda-data").hexdigest()
_ACCESS_POLICY_SHA256 = hashlib.sha256(b"cuda-access").hexdigest()
_TRAIN_TOKEN_SHA256 = hashlib.sha256(b"cuda-train-tokens").hexdigest()
_DIRECT_CERTIFICATE_SHA256 = hashlib.sha256(b"cuda-a0-certificate").hexdigest()
_MAXIMUM_CHECKPOINT_BYTES = 256 * 1024 * 1024
_REPO_ROOT = Path(__file__).resolve().parents[2]
_A5_ENDPOINT_CONFIG_ID = (
    "h6-a5-structured-parent-specific-prefix-exact-complete-latent-smoothing-v2"
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _schedule(
    matching_set: H6MatchingSetV3,
    runtime: H6PredictionRuntimeIdentity,
    estimator: H6RecognitionEstimatorSpec,
) -> H6TrainingScheduleV3:
    return H6TrainingScheduleV3.create(
        outer=H6OuterSchedule.create(
            optimizer_policy_sha256=H6_ADAMW_POLICY.optimizer_policy_sha256,
        ),
        endpoint_phases=tuple(
            H6ArmPhaseSchedule.create(
                endpoint_config_sha256=config.config_sha256,
                latent_enabled=config.latent_enabled,
                phases=(
                    (
                        TrainingPhase.RECOGNITION_ADAMW,
                        TrainingPhase.IMMUTABLE_DETACHED_SNAPSHOT,
                        TrainingPhase.MODEL_ADAMW,
                    )
                    if config.latent_enabled
                    else (TrainingPhase.MODEL_CE_ADAMW,)
                ),
            )
            for config in matching_set.endpoint_configs
        ),
        estimator=estimator,
        runtime=runtime,
    )


def _resolved_config(
    *,
    matching_set: H6MatchingSetV3,
    artifact_root: Path,
    runtime: H6PredictionRuntimeIdentity,
) -> H6PredictionV3ResolvedConfig:
    estimator = H6RecognitionEstimatorSpec.create()
    schedule = _schedule(matching_set, runtime, estimator)
    protocol = EndpointSmcProtocol.create(
        particle_counts=(128, 256, 512, 1024),
        replicate_count=64,
        registry_root_seed=2026072198,
        common_stream_domain="h6-wt2-endpoint-mc-v1",
        simultaneous_interval_count=352,
        familywise_alpha=0.01,
        critical_value_df63=4.5144904535377144,
        remainder_contraction=0.75,
    )
    objective_gate = ObjectiveGateSpec.create()
    estimator_payload = estimator.canonical_payload()
    estimator_payload["estimator_sha256"] = estimator.estimator_sha256
    runtime_payload = runtime.canonical_payload()
    runtime_payload["cuda_compute_capability"] = list(runtime.cuda_compute_capability)
    runtime_payload["runtime_identity_sha256"] = runtime.runtime_identity_sha256
    raw = {
        "schema_version": "h6-prediction-config-v3",
        "operation": "H6-Prediction",
        "source": {
            "git_head": _GIT_HEAD,
            "dirty_digest": _DIRTY_DIGEST,
            "source_sha256": _digest("cuda-source"),
        },
        "data": {
            "schema_version": "h6-data-config-v1",
            "source_url": (
                "https://s3.amazonaws.com/research.metamind.io/"
                "wikitext/wikitext-2-raw-v1.zip"
            ),
            "max_archive_bytes": 16_777_216,
            "member_paths": [
                "wikitext-2-raw/",
                "wikitext-2-raw/wiki.train.raw",
                "wikitext-2-raw/wiki.valid.raw",
                "wikitext-2-raw/wiki.test.raw",
            ],
            "allowed_compression_methods": [0, 8],
            "max_member_bytes": 16_777_216,
            "max_total_uncompressed_bytes": 33_554_432,
            "max_compression_ratio": 100,
            "observed_archive": None,
        },
        "prerequisites": {
            "correctness_manifests": {
                "H1": _digest("cuda-H1"),
                "H2": _digest("cuda-H2"),
                "H3": _digest("cuda-H3"),
                "H5": _digest("cuda-H5"),
            },
            "h1_prefix_prior_manifest_sha256": _digest("cuda-h1-prefix"),
            "h1_prefix_prior_generative_factor_schema_sha256": (
                H1_PREFIX_PRIOR_V2_GENERATIVE_FACTOR_SCHEMA_SHA256
            ),
            "smc_validation_manifest_sha256": _digest("cuda-smc-validation"),
            "prefix_certificate_set_sha256": _digest("cuda-prefix-certificates"),
            "a0_direct_exact_prefix_certificate_sha256": (_DIRECT_CERTIFICATE_SHA256),
        },
        "h5_update_binding_sha256": _digest("cuda-h5-update"),
        "training_schedule": {
            "schedule_schema": schedule.schedule_schema,
            "outer": {
                "schedule_schema": schedule.outer.schedule_schema,
                "optimizer_class": schedule.outer.optimizer_class,
                "optimizer_policy_sha256": (schedule.outer.optimizer_policy_sha256),
                "model_updates_per_batch": (schedule.outer.model_updates_per_batch),
                "validation_twentieths_per_pass": (
                    schedule.outer.validation_twentieths_per_pass
                ),
                "full_passes": schedule.outer.full_passes,
            },
            "endpoint_phases": [
                {
                    "endpoint_config_sha256": phase.endpoint_config_sha256,
                    "latent_enabled": phase.latent_enabled,
                    "phases": [item.value for item in phase.phases],
                    "recognition_updates_per_batch": (
                        phase.recognition_updates_per_batch
                    ),
                    "model_updates_per_batch": phase.model_updates_per_batch,
                    "no_op_phases": phase.no_op_phases,
                }
                for phase in schedule.endpoint_phases
            ],
            "recognition_estimator_sha256": estimator.estimator_sha256,
            "runtime_identity_sha256": runtime.runtime_identity_sha256,
            "training_noise_domain": schedule.training_noise_domain,
            "counter_mapping_sha256": H6_COUNTER_MAPPING_SHA256,
            "phase_ownership_sha256": H6_PHASE_OWNERSHIP_SHA256,
            "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        },
        "critical_values_sha256": CRITICAL_VALUES_PROTOCOL_SHA256,
        "endpoint_smc_protocol": {
            "protocol_schema": protocol.protocol_schema,
            "particle_counts": list(protocol.particle_counts),
            "replicate_count": protocol.replicate_count,
            "registry_root_seed": protocol.registry_root_seed,
            "common_stream_domain": protocol.common_stream_domain,
            "simultaneous_interval_count": protocol.simultaneous_interval_count,
            "familywise_alpha": protocol.familywise_alpha,
            "critical_value_df63": protocol.critical_value_df63,
            "remainder_contraction": protocol.remainder_contraction,
        },
        "smc_bias_semantics_sha256": SMC_BIAS_SEMANTICS.semantics_sha256,
        "attribution_matrix_sha256": arm_matrix_sha256(ARM_MATRIX_ROWS),
        "matching_policy_schema": "h6-amended-matching-policy-v3",
        "matching_policy_sha256": H6_MATCHING_POLICY_V3.policy_sha256,
        "matching_set_schema": "h6-amended-matching-set-v3",
        "matching_set_sha256": matching_set.matching_set_sha256,
        "objective_gate": {
            "schema_version": objective_gate.schema_version,
            "complete_arm_id": objective_gate.complete_arm_id,
            "emission_arm_id": objective_gate.emission_arm_id,
            "orientation": objective_gate.orientation,
            "delta_obj": objective_gate.delta_obj,
            "opening_policy": objective_gate.opening_policy,
            "evaluation_order": objective_gate.evaluation_order,
            "spec_sha256": objective_gate.spec_sha256,
        },
        "data_identity_sha256": _DATA_IDENTITY_SHA256,
        "access_policy_sha256": _ACCESS_POLICY_SHA256,
        "recognition_contract": {
            "trajectory_schema": "h6-language-recognition-trajectory-v3",
            "categorical_posterior_schema": ("h6-categorical-source-posterior-v3"),
            "terminal_mixture_schema": "h6-terminal-source-mixture-v1",
            "estimator": estimator_payload,
        },
        "runtime": runtime_payload,
        "counter_mapping_sha256": H6_COUNTER_MAPPING_SHA256,
        "phase_ownership_sha256": H6_PHASE_OWNERSHIP_SHA256,
        "checkpoint_codec_sha256": H6_CHECKPOINT_CODEC_SHA256,
        "scoring_inventory_sha256": H6_SCORING_INVENTORY_SHA256,
        "expected_test_row_count": 4104,
        "artifact_root": str(artifact_root.resolve()),
    }
    return resolve_h6_prediction_v3_config(raw, repo_root=_REPO_ROOT)


@dataclass(frozen=True)
class _CudaFixtureCase:
    runtime: H6InstalledRuntimeBindingV3
    config: H6PredictionV3ResolvedConfig
    matching_set: H6MatchingSetV3
    readiness: H6PredictionV3ReadinessToken
    plan: H6ExperimentPlanV3
    authorities: H6PredictionV3Authorities
    training_data: H6TrainingDataV3


@pytest.fixture(scope="module")
def _installed_runtime_v3() -> H6InstalledRuntimeBindingV3:
    executable = Path(sys.executable).resolve().as_posix().lower()
    assert executable == "c:/anaconda/python.exe"
    expected = H6PredictionRuntimeIdentity.create(
        python_version=platform.python_version(),
        torch_full_version=str(torch.__version__),
        cuda_runtime_version=str(torch.version.cuda or ""),
        cuda_device_name="NVIDIA GeForce RTX 5090",
        cuda_compute_capability=(12, 0),
    )
    # This is intentionally the first CUDA query or tensor-producing call in
    # the module. The runtime seam binds CUBLAS before it touches CUDA.
    return configure_installed_runtime_v3(expected_identity=expected)


@pytest.fixture(scope="module")
def _cuda_fixture_case(
    _installed_runtime_v3: H6InstalledRuntimeBindingV3,
    tmp_path_factory: pytest.TempPathFactory,
) -> _CudaFixtureCase:
    runtime = _installed_runtime_v3
    vocabulary = ByteTokenizerV1().vocabulary_identity
    matching_set = build_h6_matching_set_v3(
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
        train_token_count=2,
        train_token_sha256=_TRAIN_TOKEN_SHA256,
        vocabulary=vocabulary,
        horizon=32,
    )
    config = _resolved_config(
        matching_set=matching_set,
        artifact_root=tmp_path_factory.mktemp("h6-v3-cuda-authority").resolve(),
        runtime=runtime.identity,
    )
    readiness = _derive_h6_prediction_readiness_v3(
        config=config,
        matching_set=matching_set,
        git_head=_GIT_HEAD,
        dirty_digest=_DIRTY_DIGEST,
    )
    plan = plan_h6_experiment_v3(
        readiness=readiness,
        matching_set=matching_set,
        training_schedule=config.training_schedule,
        runtime_identity=config.runtime,
    )
    authorities = H6PredictionV3Authorities.create(
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
    )
    training_data = H6TrainingDataV3(
        data_identity_sha256=_DATA_IDENTITY_SHA256,
        readiness_sha256=readiness.readiness_sha256,
        plan_sha256=plan.plan_sha256,
        matching_set_sha256=matching_set.matching_set_sha256,
        runtime_identity_sha256=runtime.identity.runtime_identity_sha256,
        windows=build_causal_windows((2, 3), split="train"),
        vocabulary=vocabulary,
    )
    return _CudaFixtureCase(
        runtime=runtime,
        config=config,
        matching_set=matching_set,
        readiness=readiness,
        plan=plan,
        authorities=authorities,
        training_data=training_data,
    )


def _planned_attempt(
    case: _CudaFixtureCase,
    endpoint_config_id: str,
) -> H6PlannedAttemptV3:
    return next(
        attempt
        for attempt in case.plan.tuning_attempts
        if attempt.endpoint_config_id == endpoint_config_id
    )


def _executable(
    case: _CudaFixtureCase,
    endpoint_config_id: str,
) -> H6ExecutableAttemptV3:
    return bind_h6_executable_attempt_v3(
        authorities=case.authorities,
        planned_attempt=_planned_attempt(case, endpoint_config_id),
    )


def _module_state_sha256(module: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        raw = tensor.detach().to(device="cpu").contiguous().numpy().tobytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(raw)
    return hasher.hexdigest()


def _next_prediction_sha256(
    model: nn.Module,
    executable: H6ExecutableAttemptV3,
) -> str:
    prefix = CausalPrefix.create(
        receiver_t=1,
        vocabulary=executable.endpoint_config.vocabulary,
        token_ids=torch.empty(0, dtype=torch.int64, device="cpu"),
    )
    log_probs = model.prefix_log_probs(prefix)  # type: ignore[attr-defined]
    assert log_probs.device.type == "cuda"
    return hashlib.sha256(
        log_probs.detach().to(device="cpu").contiguous().numpy().tobytes()
    ).hexdigest()


def _interrupt_after_boundary(
    monkeypatch: pytest.MonkeyPatch,
    boundary_kind: str,
) -> None:
    original = attempt_v3._persist_attempt_boundary_v3
    interrupted = False

    def persist_then_interrupt(**kwargs: object) -> object:
        nonlocal interrupted
        persisted = original(**kwargs)
        if not interrupted and kwargs["boundary_kind"] == boundary_kind:
            interrupted = True
            raise RuntimeError(f"simulated CUDA loss after {boundary_kind}")
        return persisted

    monkeypatch.setattr(
        attempt_v3,
        "_persist_attempt_boundary_v3",
        persist_then_interrupt,
    )


def test_cuda_a0_uninterrupted_and_resume_are_byte_identical(
    _cuda_fixture_case: _CudaFixtureCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _cuda_fixture_case
    executable = _executable(case, "h6-a0-transformer-v2")
    uninterrupted_path = (tmp_path / "a0-uninterrupted.h6v3").resolve()
    resumed_path = (tmp_path / "a0-resumed.h6v3").resolve()
    uninterrupted = attempt_v3.execute_h6_training_attempt_v3(
        executable=executable,
        training_data=case.training_data,
        runtime=case.runtime,
        checkpoint_path=uninterrupted_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    uninterrupted_recovery = attempt_v3.recover_h6_training_attempt_v3(
        executable=executable,
        runtime=case.runtime,
        checkpoint_path=uninterrupted_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert uninterrupted_recovery is not None
    assert all(
        parameter.device.type == "cuda"
        for parameter in uninterrupted_recovery.model.parameters()
    )

    _interrupt_after_boundary(monkeypatch, "terminal")
    with pytest.raises(
        RuntimeError,
        match="simulated CUDA loss after terminal",
    ):
        attempt_v3.execute_h6_training_attempt_v3(
            executable=executable,
            training_data=case.training_data,
            runtime=case.runtime,
            checkpoint_path=resumed_path,
            maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
        )
    process_loss_recovery = attempt_v3.recover_h6_training_attempt_v3(
        executable=executable,
        runtime=case.runtime,
        checkpoint_path=resumed_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert process_loss_recovery is not None
    assert process_loss_recovery.boundary.boundary_kind == "terminal"
    resumed = attempt_v3.execute_h6_training_attempt_v3(
        executable=executable,
        training_data=case.training_data,
        runtime=case.runtime,
        checkpoint_path=resumed_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    resumed_recovery = attempt_v3.recover_h6_training_attempt_v3(
        executable=executable,
        runtime=case.runtime,
        checkpoint_path=resumed_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert resumed_recovery is not None

    assert resumed_path.read_bytes() == uninterrupted_path.read_bytes()
    assert resumed.checkpoint_bytes_sha256 == (uninterrupted.checkpoint_bytes_sha256)
    assert resumed.terminal_checkpoint == uninterrupted.terminal_checkpoint
    assert resumed.terminal_history == uninterrupted.terminal_history
    assert resumed.metric_history_sha256 == uninterrupted.metric_history_sha256
    assert resumed.validation_history_sha256 == (
        uninterrupted.validation_history_sha256
    )
    assert _module_state_sha256(resumed_recovery.model) == (
        _module_state_sha256(uninterrupted_recovery.model)
    )
    assert _next_prediction_sha256(
        resumed_recovery.model,
        executable,
    ) == _next_prediction_sha256(
        uninterrupted_recovery.model,
        executable,
    )


def test_cuda_a5_recognition_snapshot_model_and_resume_ownership(
    _cuda_fixture_case: _CudaFixtureCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _cuda_fixture_case
    executable = _executable(case, _A5_ENDPOINT_CONFIG_ID)
    uninterrupted_path = (tmp_path / "a5-uninterrupted.h6v3").resolve()
    resumed_path = (tmp_path / "a5-resumed.h6v3").resolve()
    uninterrupted = attempt_v3.execute_h6_training_attempt_v3(
        executable=executable,
        training_data=case.training_data,
        runtime=case.runtime,
        checkpoint_path=uninterrupted_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    uninterrupted_recovery = attempt_v3.recover_h6_training_attempt_v3(
        executable=executable,
        runtime=case.runtime,
        checkpoint_path=uninterrupted_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert uninterrupted_recovery is not None
    assert uninterrupted_recovery.recognition is not None

    _interrupt_after_boundary(monkeypatch, "post_recognition")
    with pytest.raises(
        RuntimeError,
        match="simulated CUDA loss after post_recognition",
    ):
        attempt_v3.execute_h6_training_attempt_v3(
            executable=executable,
            training_data=case.training_data,
            runtime=case.runtime,
            checkpoint_path=resumed_path,
            maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
        )
    phase_recovery = attempt_v3.recover_h6_training_attempt_v3(
        executable=executable,
        runtime=case.runtime,
        checkpoint_path=resumed_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert phase_recovery is not None
    assert phase_recovery.boundary.boundary_kind == "post_recognition"
    assert phase_recovery.cursor.next_phase is TrainingPhase.MODEL_ADAMW
    assert phase_recovery.resume_state is not None
    assert phase_recovery.recognition is not None
    assert all(
        parameter.device.type == "cuda"
        for parameter in phase_recovery.model.parameters()
    )
    assert all(
        parameter.device.type == "cuda"
        for parameter in phase_recovery.recognition.parameters()
    )
    snapshot = phase_recovery.checkpoint.detached_batch_snapshot
    assert snapshot is not None
    assert snapshot.snapshot_sha256 == (
        phase_recovery.resume_state.snapshot.snapshot_sha256
    )
    for name in snapshot:
        tensor = snapshot[name]
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
        assert tensor.grad_fn is None

    _built, initial_model, initial_recognition = attempt_v3._fresh_training_modules_v3(
        executable=executable,
        runtime=case.runtime,
    )
    assert initial_recognition is not None
    assert _module_state_sha256(phase_recovery.model) == (
        _module_state_sha256(initial_model)
    )
    assert _module_state_sha256(phase_recovery.recognition) != (
        _module_state_sha256(initial_recognition)
    )
    assert phase_recovery.model_optimizer.state == {}
    assert phase_recovery.recognition_optimizer is not None
    assert phase_recovery.recognition_optimizer.state

    resumed = attempt_v3.execute_h6_training_attempt_v3(
        executable=executable,
        training_data=case.training_data,
        runtime=case.runtime,
        checkpoint_path=resumed_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    resumed_recovery = attempt_v3.recover_h6_training_attempt_v3(
        executable=executable,
        runtime=case.runtime,
        checkpoint_path=resumed_path,
        maximum_checkpoint_bytes=_MAXIMUM_CHECKPOINT_BYTES,
    )
    assert resumed_recovery is not None
    assert resumed_recovery.recognition is not None

    assert resumed_path.read_bytes() == uninterrupted_path.read_bytes()
    assert resumed.checkpoint_bytes_sha256 == (uninterrupted.checkpoint_bytes_sha256)
    assert resumed.terminal_checkpoint == uninterrupted.terminal_checkpoint
    assert resumed.terminal_history == uninterrupted.terminal_history
    assert resumed.metric_history_sha256 == uninterrupted.metric_history_sha256
    assert resumed.validation_history_sha256 == (
        uninterrupted.validation_history_sha256
    )
    assert _module_state_sha256(resumed_recovery.model) == (
        _module_state_sha256(uninterrupted_recovery.model)
    )
    assert _module_state_sha256(resumed_recovery.recognition) == (
        _module_state_sha256(uninterrupted_recovery.recognition)
    )
