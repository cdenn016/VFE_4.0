"""Lazy, fail-closed runtime binding for executable H6 v3 training.

Importing this module does not configure Torch or initialize CUDA.  Production
callers must explicitly bind the already-installed Torch runtime before they
construct an optimizer.  Bounded tests may instead inject the distinct
synthetic CPU binding, which can never authorize a production campaign.
"""

from __future__ import annotations

import copy
import os
import platform
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from vfe4.predictive import canonical_model_state_sha256
from vfe4.types.h6_prediction_v3 import (
    H6_DETERMINISTIC_POLICY_DESCRIPTOR,
    H6PredictionRuntimeIdentity,
)


_REQUIRED_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _compute_capability(
    value: object,
    *,
    optional: bool,
) -> tuple[int, int] | None:
    if value is None and optional:
        return None
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError("CUDA compute capability must be an integer pair")
    return value


def _normalized_executable(value: str) -> str:
    return value.replace("\\", "/")


@dataclass(frozen=True, slots=True)
class H6LiveDeterminismSettingsV3:
    """Exact live Torch switches required by the H6 CUDA policy."""

    cublas_workspace_config: str
    deterministic_algorithms: bool
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

    def __post_init__(self) -> None:
        if type(self.cublas_workspace_config) is not str:
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be an exact string")
        for name in tuple(self.__dataclass_fields__)[1:]:
            _exact_bool(getattr(self, name), name)

    def canonical_payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def assert_frozen_policy(self) -> None:
        """Refuse every live setting that differs from the frozen policy."""

        descriptor_payload = {
            name: getattr(self, name)
            for name in H6_DETERMINISTIC_POLICY_DESCRIPTOR
            if name != "schema_version"
        }
        expected_payload = {
            name: value
            for name, value in H6_DETERMINISTIC_POLICY_DESCRIPTOR.items()
            if name != "schema_version"
        }
        if descriptor_payload != expected_payload or self.cudnn_sdp is not False:
            raise ValueError(
                "live Torch settings do not match the H6 deterministic policy"
            )


@dataclass(frozen=True, slots=True)
class H6RuntimeObservationV3:
    """Captured runtime facts, kept separate from the expected identity."""

    python_executable: str
    python_version: str
    torch_full_version: str
    cuda_runtime_version: str
    cuda_available: bool
    cuda_device_count: int
    training_device: str
    training_dtype: str
    cuda_device_name: str
    cuda_compute_capability: tuple[int, int] | None
    cublas_configured_before_cuda: bool
    deterministic_operation_probe_passed: bool
    settings: H6LiveDeterminismSettingsV3

    def __post_init__(self) -> None:
        for name in (
            "python_executable",
            "python_version",
            "torch_full_version",
            "cuda_runtime_version",
            "training_device",
            "training_dtype",
        ):
            if type(getattr(self, name)) is not str:
                raise ValueError(f"{name} must be an exact string")
        if type(self.cuda_device_name) is not str:
            raise ValueError("cuda_device_name must be an exact string")
        for name in (
            "cuda_available",
            "cublas_configured_before_cuda",
            "deterministic_operation_probe_passed",
        ):
            _exact_bool(getattr(self, name), name)
        if type(self.cuda_device_count) is not int or self.cuda_device_count < 0:
            raise ValueError("cuda_device_count must be a nonnegative integer")
        _compute_capability(
            self.cuda_compute_capability,
            optional=True,
        )
        if type(self.settings) is not H6LiveDeterminismSettingsV3:
            raise ValueError("settings must be exact H6LiveDeterminismSettingsV3")
        self.settings.__post_init__()


@dataclass(frozen=True, slots=True)
class H6SyntheticCpuRuntimeV3:
    """Distinct, bounded unit-test fixture that cannot authorize production."""

    fixture_id: str
    schema_version: Literal["h6-synthetic-cpu-runtime-v3"] = (
        "h6-synthetic-cpu-runtime-v3"
    )
    training_device: Literal["cpu"] = "cpu"
    training_dtype: Literal["float64"] = "float64"
    production_authorized: Literal[False] = False

    def __post_init__(self) -> None:
        _nonempty_string(self.fixture_id, "fixture_id")
        if (
            self.schema_version != "h6-synthetic-cpu-runtime-v3"
            or self.training_device != "cpu"
            or self.training_dtype != "float64"
            or self.production_authorized is not False
        ):
            raise ValueError("synthetic runtime is not the bounded CPU fixture")

    def assert_production_authorized(self) -> None:
        self.__post_init__()
        raise RuntimeError(
            "the bounded synthetic CPU runtime cannot authorize production"
        )


@dataclass(frozen=True, slots=True)
class H6InstalledRuntimeBindingV3:
    """Validated production binding to the already-installed Torch runtime."""

    identity: H6PredictionRuntimeIdentity
    settings: H6LiveDeterminismSettingsV3
    training_device: Literal["cuda:0"] = "cuda:0"
    training_dtype: Literal["float64"] = "float64"
    production_authorized: Literal[True] = True

    def __post_init__(self) -> None:
        if type(self.identity) is not H6PredictionRuntimeIdentity:
            raise ValueError("identity must be an exact H6PredictionRuntimeIdentity")
        self.identity.__post_init__()
        if type(self.settings) is not H6LiveDeterminismSettingsV3:
            raise ValueError("settings must be exact H6LiveDeterminismSettingsV3")
        self.settings.assert_frozen_policy()
        if (
            self.training_device != "cuda:0"
            or self.training_dtype != "float64"
            or self.production_authorized is not True
        ):
            raise ValueError("installed runtime binding changed device policy")

    def assert_production_authorized(self) -> None:
        self.__post_init__()


H6RuntimeBindingV3 = H6InstalledRuntimeBindingV3 | H6SyntheticCpuRuntimeV3


@dataclass(frozen=True, slots=True)
class H6PreparedTrainingModuleV3:
    """A copied training module moved only after canonical CPU hashing."""

    module: nn.Module
    canonical_cpu_state_sha256: str
    training_device: Literal["cuda:0", "cpu"]
    production_authorized: bool
    runtime_identity_sha256: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.module, nn.Module):
            raise ValueError("module must be a torch.nn.Module")
        if (
            type(self.canonical_cpu_state_sha256) is not str
            or len(self.canonical_cpu_state_sha256) != 64
        ):
            raise ValueError("canonical_cpu_state_sha256 must be a SHA-256 digest")
        if self.training_device not in ("cuda:0", "cpu"):
            raise ValueError("prepared training device is unsupported")
        _exact_bool(self.production_authorized, "production_authorized")
        if self.production_authorized:
            if (
                self.training_device != "cuda:0"
                or type(self.runtime_identity_sha256) is not str
                or len(self.runtime_identity_sha256) != 64
            ):
                raise ValueError("production module lacks its CUDA runtime identity")
        elif self.runtime_identity_sha256 is not None:
            raise ValueError(
                "synthetic module cannot carry a production runtime identity"
            )


def installed_torch_module_v3() -> ModuleType:
    """Return the imported Torch module without installing or selecting one."""

    return torch


def bounded_synthetic_cpu_runtime_v3(
    *,
    fixture_id: str,
) -> H6SyntheticCpuRuntimeV3:
    """Create the only CPU training binding accepted by bounded unit tests."""

    return H6SyntheticCpuRuntimeV3(fixture_id=fixture_id)


def validate_runtime_observation_v3(
    *,
    expected_identity: H6PredictionRuntimeIdentity,
    observation: H6RuntimeObservationV3,
) -> H6PredictionRuntimeIdentity:
    """Match captured facts to the frozen runtime identity or fail closed."""

    if type(expected_identity) is not H6PredictionRuntimeIdentity:
        raise ValueError(
            "expected_identity must be an exact H6PredictionRuntimeIdentity"
        )
    expected_identity.__post_init__()
    if type(observation) is not H6RuntimeObservationV3:
        raise ValueError("observation must be an exact H6RuntimeObservationV3")
    observation.__post_init__()

    if not observation.cuda_available or observation.cuda_device_count < 1:
        raise RuntimeError("CUDA is unavailable for H6 v3 production training")
    if observation.training_device != "cuda:0":
        raise ValueError("H6 v3 production training requires cuda:0")
    if observation.training_dtype != "float64":
        raise ValueError("H6 v3 production training requires float64")
    if not observation.cublas_configured_before_cuda:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG was not bound before CUDA initialization"
        )
    observation.settings.assert_frozen_policy()
    if not observation.deterministic_operation_probe_passed:
        raise RuntimeError(
            "required CUDA float64 operation has no supported deterministic "
            "implementation"
        )

    if (
        _normalized_executable(observation.python_executable).casefold()
        != expected_identity.python_executable.casefold()
    ):
        raise RuntimeError("Python executable runtime drift")
    if observation.python_version != expected_identity.python_version:
        raise RuntimeError("Python version runtime drift")
    if observation.torch_full_version != expected_identity.torch_full_version:
        raise RuntimeError("Torch runtime drift")
    if observation.cuda_runtime_version != expected_identity.cuda_runtime_version:
        raise RuntimeError("CUDA runtime drift")
    if observation.cuda_device_name != expected_identity.cuda_device_name:
        raise RuntimeError("CUDA device-name drift")
    if observation.cuda_compute_capability != expected_identity.cuda_compute_capability:
        raise RuntimeError("CUDA compute capability drift")

    observed_identity = H6PredictionRuntimeIdentity.create(
        python_version=observation.python_version,
        torch_full_version=observation.torch_full_version,
        cuda_runtime_version=observation.cuda_runtime_version,
        cuda_device_name=observation.cuda_device_name,
        cuda_compute_capability=observation.cuda_compute_capability,
    )
    if observed_identity != expected_identity:
        raise RuntimeError("H6 runtime identity drift")
    return expected_identity


def _required_backend_attribute(owner: object, name: str) -> object:
    if not hasattr(owner, name):
        raise RuntimeError(
            f"installed Torch lacks required deterministic setting {name}"
        )
    return getattr(owner, name)


def _set_required_backend_attribute(
    owner: object,
    name: str,
    value: bool,
) -> None:
    _required_backend_attribute(owner, name)
    try:
        setattr(owner, name, value)
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            f"installed Torch cannot set deterministic setting {name}"
        ) from exc


def _configure_live_determinism_v3() -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False

    matmul = torch.backends.cuda.matmul
    _set_required_backend_attribute(matmul, "allow_tf32", False)
    _set_required_backend_attribute(
        matmul,
        "allow_fp16_reduced_precision_reduction",
        False,
    )
    _set_required_backend_attribute(
        matmul,
        "allow_bf16_reduced_precision_reduction",
        False,
    )
    _set_required_backend_attribute(
        matmul,
        "allow_fp16_accumulation",
        False,
    )

    for name, enabled in (
        ("enable_flash_sdp", False),
        ("enable_mem_efficient_sdp", False),
        ("enable_cudnn_sdp", False),
        ("enable_math_sdp", True),
    ):
        setter = _required_backend_attribute(torch.backends.cuda, name)
        if not callable(setter):
            raise RuntimeError(
                f"installed Torch deterministic setting {name} is not callable"
            )
        setter(enabled)


def _capture_live_settings_v3() -> H6LiveDeterminismSettingsV3:
    matmul = torch.backends.cuda.matmul

    def backend_bool(owner: object, name: str) -> bool:
        value = _required_backend_attribute(owner, name)
        if type(value) is not bool:
            raise RuntimeError(f"installed Torch setting {name} is not an exact bool")
        return value

    def enabled_query(name: str) -> bool:
        query = _required_backend_attribute(torch.backends.cuda, name)
        if not callable(query):
            raise RuntimeError(f"installed Torch setting query {name} is not callable")
        value = query()
        if type(value) is not bool:
            raise RuntimeError(
                f"installed Torch setting query {name} returned non-bool"
            )
        return value

    return H6LiveDeterminismSettingsV3(
        cublas_workspace_config=os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG",
            "",
        ),
        deterministic_algorithms=(torch.are_deterministic_algorithms_enabled()),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cuda_matmul_allow_tf32=backend_bool(matmul, "allow_tf32"),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cuda_matmul_allow_fp16_reduced_precision_reduction=backend_bool(
            matmul,
            "allow_fp16_reduced_precision_reduction",
        ),
        cuda_matmul_allow_bf16_reduced_precision_reduction=backend_bool(
            matmul,
            "allow_bf16_reduced_precision_reduction",
        ),
        cuda_matmul_allow_fp16_accumulation=backend_bool(
            matmul,
            "allow_fp16_accumulation",
        ),
        flash_sdp=enabled_query("flash_sdp_enabled"),
        memory_efficient_sdp=enabled_query("mem_efficient_sdp_enabled"),
        cudnn_sdp=enabled_query("cudnn_sdp_enabled"),
        math_sdp=enabled_query("math_sdp_enabled"),
    )


def _deterministic_float64_probe_v3() -> None:
    try:
        query = (
            torch.linspace(
                -0.75,
                0.75,
                24,
                device="cuda:0",
                dtype=torch.float64,
            )
            .reshape(1, 2, 3, 4)
            .requires_grad_(True)
        )
        key = query.detach().clone().requires_grad_(True)
        value = torch.flip(query.detach(), dims=(-1,)).requires_grad_(True)
        with sdpa_kernel(SDPBackend.MATH):
            result = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True,
            )
        result.square().sum().backward()
        torch.cuda.synchronize(device=0)
        gradients = (query.grad, key.grad, value.grad)
        if any(gradient is None for gradient in gradients):
            raise RuntimeError("deterministic operation omitted a gradient")
        for tensor in (result, *gradients):
            if tensor is not None and not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("deterministic operation produced nonfinite values")
    except RuntimeError as exc:
        raise RuntimeError(
            "required CUDA float64 operation has no supported deterministic "
            "implementation"
        ) from exc


def configure_installed_runtime_v3(
    *,
    expected_identity: H6PredictionRuntimeIdentity,
) -> H6InstalledRuntimeBindingV3:
    """Lazily configure and validate the existing CUDA Torch installation."""

    if type(expected_identity) is not H6PredictionRuntimeIdentity:
        raise ValueError(
            "expected_identity must be an exact H6PredictionRuntimeIdentity"
        )
    expected_identity.__post_init__()

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA was initialized before H6 bound CUBLAS_WORKSPACE_CONFIG"
        )
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _REQUIRED_CUBLAS_WORKSPACE_CONFIG
    elif existing_cublas != _REQUIRED_CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("existing CUBLAS_WORKSPACE_CONFIG conflicts with H6 policy")

    _configure_live_determinism_v3()
    settings = _capture_live_settings_v3()
    settings.assert_frozen_policy()

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    if cuda_available and cuda_device_count:
        torch.cuda.set_device(0)
        device_name = torch.cuda.get_device_name(0)
        capability: tuple[int, int] | None = torch.cuda.get_device_capability(0)
        _deterministic_float64_probe_v3()
        probe_passed = True
    else:
        device_name = ""
        capability = None
        probe_passed = False

    observation = H6RuntimeObservationV3(
        python_executable=_normalized_executable(sys.executable),
        python_version=platform.python_version(),
        torch_full_version=str(torch.__version__),
        cuda_runtime_version=str(torch.version.cuda or ""),
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        training_device="cuda:0",
        training_dtype="float64",
        cuda_device_name=device_name,
        cuda_compute_capability=capability,
        cublas_configured_before_cuda=True,
        deterministic_operation_probe_passed=probe_passed,
        settings=_capture_live_settings_v3(),
    )
    identity = validate_runtime_observation_v3(
        expected_identity=expected_identity,
        observation=observation,
    )
    return H6InstalledRuntimeBindingV3(
        identity=identity,
        settings=observation.settings,
    )


def _assert_canonical_cpu_float64(module: nn.Module) -> None:
    parameters = tuple(module.parameters())
    if not parameters:
        raise ValueError("training module must own at least one parameter")
    if any(
        parameter.device.type != "cpu" or parameter.dtype is not torch.float64
        for parameter in parameters
    ):
        raise ValueError("canonical training initialization must be CPU float64")
    for tensor in module.state_dict().values():
        if type(tensor) is not Tensor:
            raise ValueError("module state must contain exact tensors")
        if tensor.layout is not torch.strided or tensor.is_quantized:
            raise ValueError("module state must be dense and unquantized")
        if tensor.device.type != "cpu":
            raise ValueError("canonical module state must remain on CPU")
        if tensor.is_complex() or (
            tensor.is_floating_point() and tensor.dtype is not torch.float64
        ):
            raise ValueError("canonical floating module state must be real float64")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("canonical module state must be finite")


def _assert_prepared_device(
    module: nn.Module,
    *,
    training_device: str,
) -> None:
    expected_device = torch.device(training_device)
    for tensor in module.state_dict().values():
        if tensor.device != expected_device:
            raise RuntimeError("training module move exposed a device fallback")
        if tensor.is_complex() or (
            tensor.is_floating_point() and tensor.dtype is not torch.float64
        ):
            raise RuntimeError("training module move changed the float64 policy")


def prepare_training_module_v3(
    *,
    cpu_module: nn.Module,
    runtime: H6RuntimeBindingV3,
) -> H6PreparedTrainingModuleV3:
    """Hash canonical CPU state, then copy it to the bound training device."""

    if not isinstance(cpu_module, nn.Module):
        raise ValueError("cpu_module must be a torch.nn.Module")
    _assert_canonical_cpu_float64(cpu_module)
    canonical_sha256 = canonical_model_state_sha256(cpu_module)

    if type(runtime) is H6InstalledRuntimeBindingV3:
        runtime.__post_init__()
        live_settings = _capture_live_settings_v3()
        if live_settings != runtime.settings:
            raise RuntimeError("live Torch deterministic settings drift")
        training_device: Literal["cuda:0", "cpu"] = "cuda:0"
        production_authorized = True
        runtime_identity_sha256: str | None = runtime.identity.runtime_identity_sha256
    elif type(runtime) is H6SyntheticCpuRuntimeV3:
        runtime.__post_init__()
        training_device = "cpu"
        production_authorized = False
        runtime_identity_sha256 = None
    else:
        raise ValueError("runtime must be an exact H6 v3 runtime binding")

    training_module = copy.deepcopy(cpu_module)
    training_module.to(
        device=torch.device(training_device),
        dtype=torch.float64,
    )
    _assert_prepared_device(
        training_module,
        training_device=training_device,
    )
    if canonical_model_state_sha256(training_module) != canonical_sha256:
        raise RuntimeError(
            "training module move changed canonical initialization bytes"
        )
    return H6PreparedTrainingModuleV3(
        module=training_module,
        canonical_cpu_state_sha256=canonical_sha256,
        training_device=training_device,
        production_authorized=production_authorized,
        runtime_identity_sha256=runtime_identity_sha256,
    )


__all__ = [
    "H6InstalledRuntimeBindingV3",
    "H6LiveDeterminismSettingsV3",
    "H6PreparedTrainingModuleV3",
    "H6RuntimeBindingV3",
    "H6RuntimeObservationV3",
    "H6SyntheticCpuRuntimeV3",
    "bounded_synthetic_cpu_runtime_v3",
    "configure_installed_runtime_v3",
    "installed_torch_module_v3",
    "prepare_training_module_v3",
    "validate_runtime_observation_v3",
]
