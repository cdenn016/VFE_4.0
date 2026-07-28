from __future__ import annotations

import dataclasses
import os
import platform
import sys
from collections.abc import Iterator

import pytest
import torch

from vfe4.predictive import canonical_model_state_sha256
from vfe4.training.h6_runtime_v3 import (
    H6InstalledRuntimeBindingV3,
    H6LiveDeterminismSettingsV3,
    H6RuntimeObservationV3,
    bounded_synthetic_cpu_runtime_v3,
    configure_installed_runtime_v3,
    installed_torch_module_v3,
    prepare_training_module_v3,
    validate_runtime_observation_v3,
)
from vfe4.training.h6_transformer import (
    H6A0ValidationProfile,
    H6CausalTransformer,
)
from vfe4.training.h6_transformer_v3 import (
    H6TrainingCausalTransformerV3,
)
from vfe4.types import VocabularyIdentity
from vfe4.types.h6_prediction_v3 import H6PredictionRuntimeIdentity


_SHA = "a" * 64


@dataclasses.dataclass(frozen=True)
class _TorchGlobalState:
    cublas_workspace_config_present: bool
    cublas_workspace_config: str | None
    deterministic_debug_mode: int
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool
    cuda_matmul_allow_fp16_reduced_precision_reduction: bool
    cuda_matmul_allow_bf16_reduced_precision_reduction: bool
    cuda_matmul_allow_fp16_accumulation: bool
    flash_sdp: bool
    memory_efficient_sdp: bool
    cudnn_sdp: bool
    math_sdp: bool


def _capture_torch_global_state() -> _TorchGlobalState:
    matmul = torch.backends.cuda.matmul
    return _TorchGlobalState(
        cublas_workspace_config_present="CUBLAS_WORKSPACE_CONFIG" in os.environ,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_debug_mode=torch.get_deterministic_debug_mode(),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cuda_matmul_allow_tf32=bool(matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cuda_matmul_allow_fp16_reduced_precision_reduction=bool(
            matmul.allow_fp16_reduced_precision_reduction
        ),
        cuda_matmul_allow_bf16_reduced_precision_reduction=bool(
            matmul.allow_bf16_reduced_precision_reduction
        ),
        cuda_matmul_allow_fp16_accumulation=bool(matmul.allow_fp16_accumulation),
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(),
        memory_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        cudnn_sdp=torch.backends.cuda.cudnn_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
    )


def _restore_torch_global_state(state: _TorchGlobalState) -> None:
    if state.cublas_workspace_config_present:
        assert state.cublas_workspace_config is not None
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = state.cublas_workspace_config
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    torch.set_deterministic_debug_mode(state.deterministic_debug_mode)
    torch.backends.cudnn.benchmark = state.cudnn_benchmark
    torch.backends.cudnn.deterministic = state.cudnn_deterministic
    torch.backends.cudnn.allow_tf32 = state.cudnn_allow_tf32
    matmul = torch.backends.cuda.matmul
    matmul.allow_tf32 = state.cuda_matmul_allow_tf32
    matmul.allow_fp16_reduced_precision_reduction = (
        state.cuda_matmul_allow_fp16_reduced_precision_reduction
    )
    matmul.allow_bf16_reduced_precision_reduction = (
        state.cuda_matmul_allow_bf16_reduced_precision_reduction
    )
    matmul.allow_fp16_accumulation = state.cuda_matmul_allow_fp16_accumulation
    torch.backends.cuda.enable_flash_sdp(state.flash_sdp)
    torch.backends.cuda.enable_mem_efficient_sdp(state.memory_efficient_sdp)
    torch.backends.cuda.enable_cudnn_sdp(state.cudnn_sdp)
    torch.backends.cuda.enable_math_sdp(state.math_sdp)


def _mutate_torch_global_setting(name: str) -> None:
    matmul = torch.backends.cuda.matmul
    if name == "cublas_workspace_config":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    elif name == "deterministic_algorithms":
        torch.use_deterministic_algorithms(False)
    elif name == "cudnn_benchmark":
        torch.backends.cudnn.benchmark = True
    elif name == "cudnn_deterministic":
        torch.backends.cudnn.deterministic = False
    elif name == "cuda_matmul_allow_tf32":
        matmul.allow_tf32 = True
    elif name == "cudnn_allow_tf32":
        torch.backends.cudnn.allow_tf32 = True
    elif name == "cuda_matmul_allow_fp16_reduced_precision_reduction":
        matmul.allow_fp16_reduced_precision_reduction = True
    elif name == "cuda_matmul_allow_bf16_reduced_precision_reduction":
        matmul.allow_bf16_reduced_precision_reduction = True
    elif name == "cuda_matmul_allow_fp16_accumulation":
        matmul.allow_fp16_accumulation = True
    elif name == "flash_sdp":
        torch.backends.cuda.enable_flash_sdp(True)
    elif name == "memory_efficient_sdp":
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    elif name == "cudnn_sdp":
        torch.backends.cuda.enable_cudnn_sdp(True)
    elif name == "math_sdp":
        torch.backends.cuda.enable_math_sdp(False)
    else:
        raise AssertionError(f"unknown Torch setting mutation {name}")


def _installed_expected_identity() -> H6PredictionRuntimeIdentity:
    return H6PredictionRuntimeIdentity.create(
        python_version=platform.python_version(),
        torch_full_version=str(torch.__version__),
        cuda_runtime_version=str(torch.version.cuda or ""),
        cuda_device_name="NVIDIA GeForce RTX 5090",
        cuda_compute_capability=(12, 0),
    )


@pytest.fixture(scope="module")
def installed_runtime_binding() -> Iterator[H6InstalledRuntimeBindingV3]:
    assert sys.executable.replace("\\", "/").casefold() == "c:/anaconda/python.exe"
    assert not torch.cuda.is_initialized()
    original_state = _capture_torch_global_state()
    try:
        yield configure_installed_runtime_v3(
            expected_identity=_installed_expected_identity(),
        )
    finally:
        _restore_torch_global_state(original_state)


def _vocabulary() -> VocabularyIdentity:
    return VocabularyIdentity("h6-runtime-test-v1", 3, _SHA)


def _profile() -> H6A0ValidationProfile:
    return H6A0ValidationProfile.create(
        vocabulary_size=3,
        position_capacity=4,
        hidden_width=4,
    )


def _expected_identity(
    *,
    torch_full_version: str = "2.9.1+cu128",
) -> H6PredictionRuntimeIdentity:
    return H6PredictionRuntimeIdentity.create(
        python_version="3.13.5",
        torch_full_version=torch_full_version,
        cuda_runtime_version="12.8",
        cuda_device_name="NVIDIA GeForce RTX 5090",
        cuda_compute_capability=(12, 0),
    )


def _valid_settings() -> H6LiveDeterminismSettingsV3:
    return H6LiveDeterminismSettingsV3(
        cublas_workspace_config=":4096:8",
        deterministic_algorithms=True,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        cuda_matmul_allow_tf32=False,
        cudnn_allow_tf32=False,
        cuda_matmul_allow_fp16_reduced_precision_reduction=False,
        cuda_matmul_allow_bf16_reduced_precision_reduction=False,
        cuda_matmul_allow_fp16_accumulation=False,
        flash_sdp=False,
        memory_efficient_sdp=False,
        cudnn_sdp=False,
        math_sdp=True,
    )


def _valid_observation(
    *,
    settings: H6LiveDeterminismSettingsV3 | None = None,
    cuda_available: bool = True,
    torch_full_version: str = "2.9.1+cu128",
) -> H6RuntimeObservationV3:
    return H6RuntimeObservationV3(
        python_executable="C:/anaconda/python.exe",
        python_version="3.13.5",
        torch_full_version=torch_full_version,
        cuda_runtime_version="12.8",
        cuda_available=cuda_available,
        cuda_device_count=1 if cuda_available else 0,
        training_device="cuda:0",
        training_dtype="float64",
        cuda_device_name=("NVIDIA GeForce RTX 5090" if cuda_available else ""),
        cuda_compute_capability=(12, 0) if cuda_available else None,
        cublas_configured_before_cuda=True,
        deterministic_operation_probe_passed=cuda_available,
        settings=_valid_settings() if settings is None else settings,
    )


def test_cpu_reference_remains_strict_cpu_float64() -> None:
    reference = H6CausalTransformer(
        vocabulary=_vocabulary(),
        profile=_profile(),
    )
    assert {
        (parameter.device.type, parameter.dtype) for parameter in reference.parameters()
    } == {("cpu", torch.float64)}

    token_ids = torch.tensor([0, 1], dtype=torch.int64)
    assert reference.sequence_log_probs(token_ids).shape == (3, 3)

    reference.to(dtype=torch.float32)
    with pytest.raises(ValueError, match="CPU float64"):
        reference.sequence_log_probs(token_ids)


def test_v3_runtime_identity_requires_cuda0_float64_policy() -> None:
    expected = _expected_identity()
    observation = _valid_observation()
    assert (
        validate_runtime_observation_v3(
            expected_identity=expected,
            observation=observation,
        )
        == expected
    )

    cpu_observation = dataclasses.replace(
        observation,
        training_device="cpu",
    )
    with pytest.raises(ValueError, match="cuda:0"):
        validate_runtime_observation_v3(
            expected_identity=expected,
            observation=cpu_observation,
        )

    flash_settings = dataclasses.replace(
        observation.settings,
        flash_sdp=True,
        math_sdp=False,
    )
    with pytest.raises(ValueError, match="deterministic policy"):
        validate_runtime_observation_v3(
            expected_identity=expected,
            observation=dataclasses.replace(
                observation,
                settings=flash_settings,
            ),
        )


def test_v3_runtime_refuses_unavailable_or_mismatched_cuda_identity() -> None:
    expected = _expected_identity()
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        validate_runtime_observation_v3(
            expected_identity=expected,
            observation=_valid_observation(cuda_available=False),
        )

    with pytest.raises(RuntimeError, match="Torch runtime drift"):
        validate_runtime_observation_v3(
            expected_identity=expected,
            observation=_valid_observation(
                torch_full_version="2.8.0+cu128",
            ),
        )

    with pytest.raises(RuntimeError, match="compute capability drift"):
        validate_runtime_observation_v3(
            expected_identity=expected,
            observation=dataclasses.replace(
                _valid_observation(),
                cuda_compute_capability=(9, 0),
            ),
        )


def test_v3_cpu_canonical_initialization_is_stable() -> None:
    reference = H6CausalTransformer(
        vocabulary=_vocabulary(),
        profile=_profile(),
    )
    training = H6TrainingCausalTransformerV3(
        vocabulary=_vocabulary(),
        profile=_profile(),
        allow_synthetic_cpu=True,
    )
    assert tuple(reference.state_dict()) == tuple(training.state_dict())
    for name, value in reference.state_dict().items():
        assert torch.equal(value, training.state_dict()[name])

    expected_sha256 = canonical_model_state_sha256(reference)
    assert canonical_model_state_sha256(training) == expected_sha256

    synthetic = bounded_synthetic_cpu_runtime_v3(
        fixture_id="task4-unit-fixture",
    )
    with pytest.raises(RuntimeError, match="cannot authorize production"):
        synthetic.assert_production_authorized()
    prepared = prepare_training_module_v3(
        cpu_module=training,
        runtime=synthetic,
    )
    assert prepared.module is not training
    assert prepared.canonical_cpu_state_sha256 == expected_sha256
    assert prepared.training_device == "cpu"
    assert prepared.production_authorized is False
    assert canonical_model_state_sha256(prepared.module) == expected_sha256
    assert {
        (parameter.device.type, parameter.dtype) for parameter in training.parameters()
    } == {("cpu", torch.float64)}


def test_v3_runtime_never_installs_or_selects_another_torch() -> None:
    imported = torch
    imported_file = torch.__file__
    assert installed_torch_module_v3() is imported

    with pytest.raises(RuntimeError, match="Torch runtime drift"):
        validate_runtime_observation_v3(
            expected_identity=_expected_identity(
                torch_full_version="99.0.0+alternate",
            ),
            observation=_valid_observation(),
        )

    assert sys.modules["torch"] is imported
    assert torch.__file__ == imported_file


def test_v3_configures_the_real_installed_cuda_runtime(
    installed_runtime_binding: H6InstalledRuntimeBindingV3,
) -> None:
    assert type(installed_runtime_binding.identity.torch_full_version) is str
    assert installed_runtime_binding.identity.torch_full_version == str(
        torch.__version__
    )
    assert installed_runtime_binding.training_device == "cuda:0"
    assert installed_runtime_binding.training_dtype == "float64"
    assert installed_runtime_binding.production_authorized is True
    assert installed_runtime_binding.settings == _valid_settings()


@pytest.mark.parametrize(
    "setting_name",
    (
        "cublas_workspace_config",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_fp16_reduced_precision_reduction",
        "cuda_matmul_allow_bf16_reduced_precision_reduction",
        "cuda_matmul_allow_fp16_accumulation",
        "flash_sdp",
        "memory_efficient_sdp",
        "cudnn_sdp",
        "math_sdp",
    ),
)
def test_v3_prepare_refuses_live_deterministic_setting_drift(
    installed_runtime_binding: H6InstalledRuntimeBindingV3,
    setting_name: str,
) -> None:
    configured_state = _capture_torch_global_state()
    training = H6TrainingCausalTransformerV3(
        vocabulary=_vocabulary(),
        profile=_profile(),
        allow_synthetic_cpu=True,
    )
    try:
        _mutate_torch_global_setting(setting_name)
        with pytest.raises(
            RuntimeError,
            match="live Torch deterministic settings drift",
        ):
            prepare_training_module_v3(
                cpu_module=training,
                runtime=installed_runtime_binding,
            )
    finally:
        _restore_torch_global_state(configured_state)
    assert _capture_torch_global_state() == configured_state
