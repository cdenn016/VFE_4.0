"""Live environment, allocation, throughput, and power evidence collectors.

The records in :mod:`vfe4.artifacts.environment` deliberately validate typed
observations without touching the host runtime.  This module owns the opposite
boundary: it measures those facts and immediately converts them into the
content-bound records consumed by readiness.  Production callers must execute
the shape-identical allocation probes in clean child processes.
"""

from __future__ import annotations

import csv
import ctypes
import hashlib
import locale
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol, TypeVar, cast

from vfe4.types.training import WT103ArmSpec, owned_sha256

from .environment import (
    AllocationObservation,
    ComponentBenchmark,
    DependencyLockIdentity,
    EnvironmentObservation,
    EnvironmentRecord,
    PowerProviderIdentity,
    ResourceComponentSpec,
    TrainingExecutionIdentity,
    capture_environment,
)


_T = TypeVar("_T")


class LiveEnvironmentProvider(Protocol):
    """Provider seam for facts captured before CUDA device initialization."""

    def capture_before_device_work(self) -> EnvironmentObservation: ...


class AllocationBackend(Protocol):
    """Minimal allocator/device seam used by a clean child probe."""

    def reset_peak_memory_stats(self, device_ordinal: int) -> None: ...

    def synchronize(self, device_ordinal: int) -> None: ...

    def peak_memory_allocated(self, device_ordinal: int) -> int: ...

    def peak_memory_reserved(self, device_ordinal: int) -> int: ...

    def device_uuid(self, device_ordinal: int) -> str: ...

    def physical_device_bytes(self, device_ordinal: int) -> int: ...

    def host_available_bytes(self) -> int: ...

    def disk_available_bytes(self, path: Path) -> int: ...


class PowerSampler(Protocol):
    """Run one timed operation while sampling board power."""

    def sample(
        self,
        operation: Callable[[], _T],
        *,
        on_observation: Callable[["PowerObservation"], None] | None = None,
    ) -> tuple[_T, tuple[float, ...]]: ...


@dataclass(frozen=True, slots=True)
class PowerObservation:
    """One timestamped board-power observation on the monotonic clock."""

    watts: float
    monotonic_ns: int

    def __post_init__(self) -> None:
        if (
            type(self.watts) is not float
            or not math.isfinite(self.watts)
            or self.watts < 0.0
            or type(self.monotonic_ns) is not int
            or self.monotonic_ns < 0
        ):
            raise ValueError("power observation is invalid")


class PowerSampleOperationFailure(RuntimeError):
    """Carry partial telemetry and the exact operation outcome on failure."""

    def __init__(
        self,
        *,
        sampling_error: BaseException | None,
        observations: tuple[PowerObservation, ...],
        operation_completed: bool,
        operation_result: object,
        operation_error: BaseException | None,
    ) -> None:
        if (
            (
                sampling_error is not None
                and not isinstance(sampling_error, BaseException)
            )
            or type(observations) is not tuple
            or any(type(item) is not PowerObservation for item in observations)
            or type(operation_completed) is not bool
            or (
                not operation_completed
                and (
                    operation_result is not None
                    or operation_error is not None
                )
            )
            or (
                operation_error is not None
                and not isinstance(operation_error, BaseException)
            )
            or (
                sampling_error is None
                and operation_error is None
            )
        ):
            raise ValueError("power-sampled operation failure is invalid")
        for item in observations:
            item.__post_init__()
        message = (
            "power sampling failed"
            if sampling_error is not None
            else "power-sampled operation failed"
        )
        super().__init__(message)
        self.sampling_error = sampling_error
        self.observations = observations
        self.operation_completed = operation_completed
        self.operation_result = operation_result
        self.operation_error = operation_error


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be an exact nonnegative int")
    return value


def _physical_ram_bytes() -> int:
    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = (
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            )

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
            ctypes.byref(status)
        ):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys)
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_PHYS_PAGES")
    return int(page_size) * int(page_count)


def _available_ram_bytes() -> int:
    if os.name == "nt":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = (
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            )

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
            ctypes.byref(status)
        ):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size) * int(page_count)


def _run_checked(
    command: tuple[str, ...],
    *,
    maximum_output_bytes: int = 1_048_576,
) -> bytes:
    completed = subprocess.run(
        command,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30.0,
    )
    if len(completed.stdout) > maximum_output_bytes:
        raise RuntimeError("live environment command output exceeded its bound")
    return completed.stdout


def _nvidia_smi_rows(executable: str) -> tuple[tuple[str, ...], ...]:
    raw = _run_checked(
        (
            executable,
            "--query-gpu=name,uuid,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    text = raw.decode("utf-8", errors="strict")
    rows = tuple(
        tuple(cell.strip() for cell in row)
        for row in csv.reader(text.splitlines())
        if row
    )
    if not rows or any(len(row) != 5 or any(not cell for cell in row) for row in rows):
        raise RuntimeError("nvidia-smi returned a malformed GPU inventory")
    return rows


class SystemLiveEnvironmentProvider:
    """Capture exact runtime facts without initializing a CUDA context."""

    def capture_before_device_work(self) -> EnvironmentObservation:
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA device work started before environment capture")
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise RuntimeError("nvidia-smi is required for live GPU identity")
        rows = _nvidia_smi_rows(executable)
        thread_values = {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        }
        thread_values.update(
            {
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
            }
        )
        blas_sha256 = owned_sha256(
            "vfe4.wt103.live-blas-config.v1",
            torch.__config__.show(),
        )
        thread_sha256 = owned_sha256(
            "vfe4.wt103.live-thread-settings.v1",
            thread_values,
        )
        cudnn_version = torch.backends.cudnn.version()
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "environment inspection unexpectedly initialized CUDA"
            )
        driver_versions = {row[4] for row in rows}
        if len(driver_versions) != 1:
            raise RuntimeError("GPU driver versions disagree across devices")
        return EnvironmentObservation(
            captured_utc=(
                datetime.now(UTC)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            ),
            device_work_started=False,
            python_version=platform.python_version(),
            pytorch_version=str(torch.__version__),
            cuda_runtime_version=str(torch.version.cuda or "not_available"),
            cudnn_version=str(cudnn_version or "not_available"),
            driver_version=next(iter(driver_versions)),
            os_name=os.name,
            platform_system=platform.system(),
            platform_release=platform.release(),
            cpu_name=platform.processor() or platform.machine(),
            logical_cpu_count=os.cpu_count() or 1,
            physical_ram_bytes=_physical_ram_bytes(),
            gpu_names=tuple(row[0] for row in rows),
            gpu_device_uuids=tuple(row[1] for row in rows),
            gpu_total_bytes=tuple(
                int(row[2]) * 1024 * 1024 for row in rows
            ),
            compute_capabilities=tuple(row[3] for row in rows),
            blas_identity_sha256=blas_sha256,
            thread_settings_sha256=thread_sha256,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
            locale_name=locale.setlocale(locale.LC_ALL, None),
            timezone_name=os.environ.get("TZ") or time.tzname[0],
        )


def capture_live_environment(
    *,
    dependency_lock: DependencyLockIdentity,
    provider: LiveEnvironmentProvider | None = None,
) -> EnvironmentRecord:
    """Capture and bind the environment before any CUDA device work."""

    if type(dependency_lock) is not DependencyLockIdentity:
        raise ValueError("dependency_lock must be exact")
    selected = provider or SystemLiveEnvironmentProvider()
    observation = selected.capture_before_device_work()
    if type(observation) is not EnvironmentObservation:
        raise ValueError("live provider returned a nonexact observation")
    return capture_environment(
        observation,
        dependency_lock=dependency_lock,
    )


class TorchCudaAllocationBackend:
    """PyTorch CUDA allocator measurements for an isolated child process."""

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for allocation preflight")
        self._torch = torch

    def reset_peak_memory_stats(self, device_ordinal: int) -> None:
        self._torch.cuda.synchronize(device_ordinal)
        self._torch.cuda.reset_peak_memory_stats(device_ordinal)

    def synchronize(self, device_ordinal: int) -> None:
        self._torch.cuda.synchronize(device_ordinal)

    def peak_memory_allocated(self, device_ordinal: int) -> int:
        return int(self._torch.cuda.max_memory_allocated(device_ordinal))

    def peak_memory_reserved(self, device_ordinal: int) -> int:
        return int(self._torch.cuda.max_memory_reserved(device_ordinal))

    def _properties(self, device_ordinal: int):
        return self._torch.cuda.get_device_properties(device_ordinal)

    def device_uuid(self, device_ordinal: int) -> str:
        value = getattr(self._properties(device_ordinal), "uuid", None)
        if value is None:
            executable = shutil.which("nvidia-smi")
            if executable is None:
                raise RuntimeError("CUDA device UUID is unavailable")
            rows = _nvidia_smi_rows(executable)
            if device_ordinal >= len(rows):
                raise RuntimeError("CUDA device ordinal is outside inventory")
            return rows[device_ordinal][1]
        return str(value)

    def physical_device_bytes(self, device_ordinal: int) -> int:
        return int(self._properties(device_ordinal).total_memory)

    def host_available_bytes(self) -> int:
        return _available_ram_bytes()

    def disk_available_bytes(self, path: Path) -> int:
        return int(shutil.disk_usage(path.resolve()).free)


def measure_shape_identical_allocation(
    *,
    spec: WT103ArmSpec,
    execution_identity: TrainingExecutionIdentity,
    environment: EnvironmentRecord,
    device_ordinal: int,
    checkpoint_duplicate_bytes: int,
    checkpoint_root: Path,
    operation: Callable[[], object],
    backend: AllocationBackend | None = None,
) -> AllocationObservation:
    """Measure one complete arm path against its captured device identity."""

    if type(spec) is not WT103ArmSpec:
        raise ValueError("spec must be exact")
    if type(execution_identity) is not TrainingExecutionIdentity:
        raise ValueError("execution_identity must be exact")
    if type(environment) is not EnvironmentRecord:
        raise ValueError("environment must be exact")
    if type(device_ordinal) is not int or device_ordinal < 0:
        raise ValueError("device_ordinal must be an exact nonnegative int")
    _exact_nonnegative_int(
        checkpoint_duplicate_bytes,
        "checkpoint_duplicate_bytes",
    )
    if not isinstance(checkpoint_root, Path):
        raise ValueError("checkpoint_root must be a Path")
    if not callable(operation):
        raise ValueError("operation must be callable")
    spec.__post_init__()
    execution_identity.__post_init__()
    environment.__post_init__()
    if execution_identity.environment_sha256 != environment.environment_sha256:
        raise ValueError("execution identity differs from captured environment")
    if device_ordinal >= len(environment.observation.gpu_device_uuids):
        raise ValueError("device ordinal is outside captured environment")
    selected = backend or TorchCudaAllocationBackend()
    measured_uuid = selected.device_uuid(device_ordinal)
    measured_total = selected.physical_device_bytes(device_ordinal)
    captured_uuid = environment.observation.gpu_device_uuids[device_ordinal]
    captured_total = environment.observation.gpu_total_bytes[device_ordinal]
    if measured_uuid != captured_uuid or measured_total != captured_total:
        raise ValueError(
            "live device identity differs from captured environment"
        )
    selected.reset_peak_memory_stats(device_ordinal)
    operation()
    selected.synchronize(device_ordinal)
    return AllocationObservation.shape_identical_for_arm(
        spec,
        execution_identity=execution_identity,
        device_ordinal=device_ordinal,
        device_uuid=captured_uuid,
        physical_device_bytes=captured_total,
        peak_device_allocated_bytes=selected.peak_memory_allocated(
            device_ordinal
        ),
        peak_device_reserved_bytes=selected.peak_memory_reserved(
            device_ordinal
        ),
        host_available_bytes=selected.host_available_bytes(),
        checkpoint_duplicate_bytes=checkpoint_duplicate_bytes,
        disk_available_bytes=selected.disk_available_bytes(checkpoint_root),
    )


def _validate_sample_result(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            "benchmark operation must return exact positive completed work units"
        )
    return value


def benchmark_live_component(
    *,
    component: ResourceComponentSpec,
    execution_identity: TrainingExecutionIdentity,
    operation: Callable[[], int],
    power_provider: PowerProviderIdentity | None,
    power_sampler: PowerSampler | None,
    synchronize: Callable[[], None] | None = None,
    monotonic: Callable[[], float] = time.perf_counter,
) -> ComponentBenchmark:
    """Run the frozen warmup/sample inventory and retain worst-case evidence."""

    if type(component) is not ResourceComponentSpec:
        raise ValueError("component must be exact")
    if type(execution_identity) is not TrainingExecutionIdentity:
        raise ValueError("execution_identity must be exact")
    component.__post_init__()
    execution_identity.__post_init__()
    if not callable(operation) or not callable(monotonic):
        raise ValueError("benchmark callables are invalid")
    if component.uses_gpu:
        if (
            type(power_provider) is not PowerProviderIdentity
            or power_sampler is None
        ):
            raise ValueError(
                "GPU benchmark requires exact 100 ms power evidence"
            )
        power_provider.__post_init__()
        if power_provider.sample_interval_ms != 100:
            raise ValueError("power sampling interval is not frozen at 100 ms")
    elif power_provider is not None or power_sampler is not None:
        raise ValueError("CPU component cannot consume GPU power evidence")

    def run_one() -> int:
        completed = _validate_sample_result(operation())
        if synchronize is not None:
            synchronize()
        return completed

    for _ in range(component.warmup_count):
        run_one()

    durations: list[float] = []
    throughputs: list[float] = []
    power_samples: list[float] = []
    for _ in range(component.sample_count):
        started = monotonic()
        if component.uses_gpu:
            assert power_sampler is not None
            completed, samples = power_sampler.sample(run_one)
            if (
                type(samples) is not tuple
                or not samples
                or any(
                    type(value) not in (int, float)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    for value in samples
                )
            ):
                raise ValueError("GPU power sample inventory is invalid")
            power_samples.extend(float(value) for value in samples)
        else:
            completed = run_one()
        duration = monotonic() - started
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("benchmark clock returned an invalid duration")
        duration_for_rate = max(duration, sys.float_info.epsilon)
        durations.append(duration)
        throughputs.append(completed / duration_for_rate)

    maximum_power: float | None
    if component.uses_gpu:
        maximum_power = max(power_samples)
    else:
        maximum_power = 0.0
    return ComponentBenchmark.observed_for(
        component,
        execution_identity=execution_identity,
        minimum_throughput_per_second=min(throughputs),
        maximum_duration_seconds=max(durations),
        maximum_board_power_watts=maximum_power,
        power_provider=power_provider,
    )


def discover_nvidia_smi_power_provider(
    *,
    device_ordinal: int = 0,
) -> tuple[PowerProviderIdentity, "NvidiaSmiPowerSampler"]:
    """Bind the installed nvidia-smi executable and its board-power limit."""

    if type(device_ordinal) is not int or device_ordinal < 0:
        raise ValueError("device_ordinal must be an exact nonnegative int")
    executable_text = shutil.which("nvidia-smi")
    if executable_text is None:
        raise RuntimeError("nvidia-smi is unavailable")
    executable = Path(executable_text).resolve(strict=True)
    version = _run_checked((str(executable), "--version")).decode(
        "utf-8",
        errors="strict",
    ).strip()
    raw_limit = _run_checked(
        (
            str(executable),
            f"--id={device_ordinal}",
            "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
        )
    ).decode("ascii", errors="strict").strip()
    limit_lines = tuple(line.strip() for line in raw_limit.splitlines() if line.strip())
    if len(limit_lines) != 1:
        raise RuntimeError("nvidia-smi returned an ambiguous power limit")
    limit = float(limit_lines[0])
    provider = PowerProviderIdentity.create(
        provider_kind="nvidia-smi",
        provider_version=version,
        provider_executable_sha256=hashlib.sha256(
            executable.read_bytes()
        ).hexdigest(),
        sample_interval_ms=100,
        reported_power_limit_watts=limit,
    )
    return provider, NvidiaSmiPowerSampler(
        executable=executable,
        device_ordinal=device_ordinal,
    )


@dataclass(frozen=True, slots=True)
class NvidiaSmiPowerSampler:
    """Sample board power every 100 ms while one operation executes."""

    executable: Path
    device_ordinal: int
    sample_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.executable, Path)
            or not self.executable.is_file()
            or self.executable.is_symlink()
            or type(self.device_ordinal) is not int
            or self.device_ordinal < 0
            or type(self.sample_interval_seconds) is not float
            or self.sample_interval_seconds != 0.1
        ):
            raise ValueError("nvidia-smi power sampler is invalid")

    def _read_power(self) -> float:
        raw = _run_checked(
            (
                str(self.executable),
                f"--id={self.device_ordinal}",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
            ),
            maximum_output_bytes=4_096,
        ).decode("ascii", errors="strict").strip()
        lines = tuple(line.strip() for line in raw.splitlines() if line.strip())
        if len(lines) != 1:
            raise RuntimeError("nvidia-smi returned ambiguous board power")
        value = float(lines[0])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("nvidia-smi returned invalid board power")
        return value

    def sample(
        self,
        operation: Callable[[], _T],
        *,
        on_observation: Callable[[PowerObservation], None] | None = None,
    ) -> tuple[_T, tuple[float, ...]]:
        if not callable(operation) or (
            on_observation is not None and not callable(on_observation)
        ):
            raise ValueError("power sample callables are invalid")
        stop = threading.Event()
        observations: list[PowerObservation] = []
        failures: list[BaseException] = []
        observation_lock = threading.Lock()
        accepting_observations = True

        def observe() -> None:
            nonlocal accepting_observations
            observation = PowerObservation(
                watts=float(self._read_power()),
                monotonic_ns=time.perf_counter_ns(),
            )
            observation.__post_init__()
            with observation_lock:
                if not accepting_observations:
                    return
                observations.append(observation)
                if on_observation is not None:
                    on_observation(observation)

        try:
            observe()
        except BaseException as exc:
            raise PowerSampleOperationFailure(
                sampling_error=exc,
                observations=tuple(observations),
                operation_completed=False,
                operation_result=None,
                operation_error=None,
            ) from exc

        def worker() -> None:
            try:
                while not stop.wait(self.sample_interval_seconds):
                    observe()
            except BaseException as exc:  # pragma: no cover - live provider failure
                failures.append(exc)
                stop.set()

        thread = threading.Thread(
            target=worker,
            name="vfe4-nvidia-smi-power-sampler",
            daemon=True,
        )
        thread.start()
        operation_completed = False
        operation_result: object = None
        operation_error: BaseException | None = None
        try:
            try:
                operation_result = operation()
            except BaseException as exc:
                operation_error = exc
            finally:
                operation_completed = True
        finally:
            stop.set()
            thread.join(timeout=5.0)
        if thread.is_alive():
            with observation_lock:
                accepting_observations = False
            failures.append(RuntimeError("power sampler did not terminate"))
        sampling_error = failures[0] if failures else None
        if sampling_error is not None or operation_error is not None:
            failure = PowerSampleOperationFailure(
                sampling_error=sampling_error,
                observations=tuple(observations),
                operation_completed=operation_completed,
                operation_result=operation_result,
                operation_error=operation_error,
            )
            raise failure from (
                sampling_error
                if sampling_error is not None
                else operation_error
            )
        return cast(_T, operation_result), tuple(
            item.watts for item in observations
        )


__all__ = [
    "AllocationBackend",
    "LiveEnvironmentProvider",
    "NvidiaSmiPowerSampler",
    "PowerObservation",
    "PowerSampleOperationFailure",
    "PowerSampler",
    "SystemLiveEnvironmentProvider",
    "TorchCudaAllocationBackend",
    "benchmark_live_component",
    "capture_live_environment",
    "discover_nvidia_smi_power_provider",
    "measure_shape_identical_allocation",
]
