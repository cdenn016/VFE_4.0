"""Truthful, fail-closed observations for production WikiText-103 metrics."""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypeVar


class ProductionObservationError(RuntimeError):
    """A production metric could not be supported by an actual observation."""


@dataclass(frozen=True, slots=True)
class ProjectedMetric:
    """One exact objective numerator, denominator, and derived value."""

    numerator: float
    denominator: int
    value: float

    def __post_init__(self) -> None:
        if (
            type(self.numerator) is not float
            or not math.isfinite(self.numerator)
            or type(self.denominator) is not int
            or self.denominator <= 0
            or type(self.value) is not float
            or not math.isfinite(self.value)
            or self.value != self.numerator / self.denominator
        ):
            raise ProductionObservationError(
                "projected metric arithmetic is invalid"
            )


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ProductionObservationError(f"{name} must be a finite float")
    return value


def _indexed_sum(
    terms: dict[str, float],
    base: str,
) -> tuple[float, tuple[str, ...]]:
    prefix = f"{base}["
    rows: list[tuple[int, str, float]] = []
    for key, value in terms.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix.endswith("]") or not suffix[:-1].isdigit():
            raise ProductionObservationError(
                f"{base} indexed term name is malformed"
            )
        rows.append((int(suffix[:-1]), key, _finite_float(value, key)))
    rows.sort(key=lambda item: item[0])
    if not rows or tuple(item[0] for item in rows) != tuple(range(len(rows))):
        raise ProductionObservationError(
            f"{base} indexed terms must be contiguous from zero"
        )
    return math.fsum(item[2] for item in rows), tuple(
        item[1] for item in rows
    )


def _project(numerator: float, denominator: int) -> ProjectedMetric:
    owned = float(numerator)
    return ProjectedMetric(
        numerator=owned,
        denominator=denominator,
        value=owned / denominator,
    )


def _same_float32_reduction(left: float, right: float) -> bool:
    """Accept only the rounding envelope of a short float32 reduction."""

    scale = max(abs(left), abs(right), 1.0)
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=8.0 * scale * 2.0**-23,
    )


def project_objective_metrics(
    *,
    objective_kind: Literal[
        "cross_entropy",
        "complete_elbo",
        "emission_only_ablation_non_elbo",
    ],
    objective_terms: dict[str, float],
    complete_elbo_numerator: float | None,
    complete_elbo_value: float | None,
    counted_targets: int,
) -> dict[str, ProjectedMetric]:
    """Project one train step without validation substitution or fake zeros."""

    if (
        type(objective_terms) is not dict
        or not objective_terms
        or type(counted_targets) is not int
        or counted_targets <= 0
    ):
        raise ProductionObservationError(
            "objective projection requires exact terms and target count"
        )
    if len(set(objective_terms)) != len(objective_terms):
        raise ProductionObservationError("objective term names are not unique")
    for key, value in objective_terms.items():
        if type(key) is not str or not key:
            raise ProductionObservationError("objective term name is invalid")
        _finite_float(value, key)

    if objective_kind == "cross_entropy":
        if (
            set(objective_terms) != {"cross_entropy_value"}
            or complete_elbo_numerator is not None
            or complete_elbo_value is not None
        ):
            raise ProductionObservationError(
                "cross-entropy objective terms differ from the exact schema"
            )
        return {
            "train_cross_entropy": _project(
                objective_terms["cross_entropy_value"],
                counted_targets,
            )
        }

    emission, emission_keys = _indexed_sum(
        objective_terms,
        "expected_log_emission",
    )
    if objective_kind == "emission_only_ablation_non_elbo":
        expected_keys = set(emission_keys) | {"emission_only_non_elbo"}
        if (
            set(objective_terms) != expected_keys
            or complete_elbo_numerator is not None
            or complete_elbo_value is not None
            or objective_terms["emission_only_non_elbo"] != emission
        ):
            raise ProductionObservationError(
                "emission-only terms differ from the exact schema"
            )
        return {
            "emission_only_non_elbo": _project(
                emission,
                counted_targets,
            )
        }

    if objective_kind != "complete_elbo":
        raise ProductionObservationError("objective kind is unknown")
    complete_numerator = _finite_float(
        complete_elbo_numerator,
        "complete_elbo_numerator",
    )
    complete_value = _finite_float(
        complete_elbo_value,
        "complete_elbo_value",
    )
    indexed_bases = (
        "model_source_cross_entropy",
        "model_transition_cross_entropy",
        "state_source_cross_entropy",
        "state_transition_cross_entropy",
        "model_source_kl",
        "state_source_kl",
    )
    indexed: dict[str, float] = {
        "expected_log_emission": emission,
    }
    indexed_keys = set(emission_keys)
    expected_indices = tuple(
        int(key.rsplit("[", 1)[1][:-1]) for key in emission_keys
    )
    for base in indexed_bases:
        total, keys = _indexed_sum(objective_terms, base)
        indices = tuple(int(key.rsplit("[", 1)[1][:-1]) for key in keys)
        if indices != expected_indices:
            raise ProductionObservationError(
                f"{base} indexed terms must match the emission horizon"
            )
        indexed[base] = total
        indexed_keys.update(keys)
    required_scalar_names = (
        "initial_model_cross_entropy",
        "initial_state_cross_entropy",
        "continuous_recognition_entropy",
        "conditional_source_entropy_estimate",
        "joint_recognition_entropy_estimate",
        "complete_elbo_numerator",
    )
    if "estimator_error_bound" in objective_terms:
        raise ProductionObservationError(
            "structured-factor estimator error bound is not applicable"
        )
    scalar_names = required_scalar_names
    if set(objective_terms) != indexed_keys | set(scalar_names):
        raise ProductionObservationError(
            "complete-ELBO terms differ from the exact schema"
        )
    conditional_source_entropy = objective_terms[
        "conditional_source_entropy_estimate"
    ]
    joint_entropy = objective_terms[
        "joint_recognition_entropy_estimate"
    ]
    expected_joint_entropy = (
        objective_terms["continuous_recognition_entropy"]
        + conditional_source_entropy
    )
    if not _same_float32_reduction(
        joint_entropy,
        expected_joint_entropy,
    ):
        raise ProductionObservationError(
            "joint recognition entropy changed its chain-rule sum"
        )
    expected_source_kl = (
        indexed["model_source_cross_entropy"]
        + indexed["state_source_cross_entropy"]
        - conditional_source_entropy
    )
    observed_source_kl = (
        indexed["model_source_kl"] + indexed["state_source_kl"]
    )
    if not _same_float32_reduction(
        observed_source_kl,
        expected_source_kl,
    ):
        raise ProductionObservationError(
            "source KL diagnostics changed cross-entropy-minus-entropy"
        )
    reconstructed = math.fsum(
        (
            emission,
            -objective_terms["initial_model_cross_entropy"],
            -objective_terms["initial_state_cross_entropy"],
            -indexed["model_source_cross_entropy"],
            -indexed["model_transition_cross_entropy"],
            -indexed["state_source_cross_entropy"],
            -indexed["state_transition_cross_entropy"],
            joint_entropy,
        )
    )
    if not _same_float32_reduction(complete_numerator, reconstructed):
        raise ProductionObservationError(
            "complete-ELBO factor reconstruction changed"
        )
    if (
        objective_terms["complete_elbo_numerator"]
        != complete_numerator
        or complete_value != complete_numerator / counted_targets
    ):
        raise ProductionObservationError(
            "complete-ELBO numerator/value arithmetic differs from StepResult"
        )
    numerators = {
        **indexed,
        "initial_model_cross_entropy": objective_terms[
            "initial_model_cross_entropy"
        ],
        "initial_state_cross_entropy": objective_terms[
            "initial_state_cross_entropy"
        ],
        "continuous_recognition_entropy": objective_terms[
            "continuous_recognition_entropy"
        ],
        "conditional_source_entropy_estimate": (
            conditional_source_entropy
        ),
        "joint_recognition_entropy_estimate": joint_entropy,
        "complete_elbo": complete_numerator,
    }
    return {
        name: _project(value, counted_targets)
        for name, value in numerators.items()
    }


@dataclass(frozen=True, slots=True)
class SourceObservation:
    """Raw categorical-source totals from the active recognition rows."""

    entropy_sum: float
    source_row_count: int
    support_size_sum: float

    def __post_init__(self) -> None:
        if (
            type(self.entropy_sum) is not float
            or not math.isfinite(self.entropy_sum)
            or self.entropy_sum < 0.0
            or type(self.source_row_count) is not int
            or self.source_row_count < 0
            or type(self.support_size_sum) is not float
            or not math.isfinite(self.support_size_sum)
            or (
                self.source_row_count == 0
                and (
                    self.entropy_sum != 0.0
                    or self.support_size_sum != 0.0
                )
            )
            or (
                self.source_row_count > 0
                and self.support_size_sum < self.source_row_count
            )
        ):
            raise ProductionObservationError(
                "source observation requires finite totals; zero rows require "
                "exactly zero totals"
            )

    @property
    def mean_entropy(self) -> float:
        if self.source_row_count == 0:
            raise ProductionObservationError(
                "source entropy is not applicable with zero source rows"
            )
        return self.entropy_sum / self.source_row_count

    @property
    def mean_support_size(self) -> float:
        if self.source_row_count == 0:
            raise ProductionObservationError(
                "source support is not applicable with zero source rows"
            )
        return self.support_size_sum / self.source_row_count

    @property
    def effective_source_count(self) -> float:
        return math.exp(self.mean_entropy)


@dataclass(frozen=True, slots=True)
class NumericalObservation:
    """Actual local SPD/solve diagnostics from the most recent train step."""

    minimum_cholesky_pivot: float
    failed_pivots: int
    condition_estimate: float
    solve_residual: float
    nonfinite_count: int

    def __post_init__(self) -> None:
        floats = (
            self.minimum_cholesky_pivot,
            self.condition_estimate,
            self.solve_residual,
        )
        if (
            any(type(value) is not float or not math.isfinite(value) for value in floats)
            or self.condition_estimate < 0.0
            or self.solve_residual < 0.0
            or type(self.failed_pivots) is not int
            or self.failed_pivots < 0
            or type(self.nonfinite_count) is not int
            or self.nonfinite_count < 0
        ):
            raise ProductionObservationError(
                "numerical observations must be finite measured values"
            )


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    """Current and peak host/device memory counters."""

    process_rss_bytes: int
    process_hwm_bytes: int
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise ProductionObservationError(
                "memory observations must be nonnegative exact bytes"
            )
        if self.process_hwm_bytes < self.process_rss_bytes:
            raise ProductionObservationError(
                "process HWM cannot be below current RSS"
            )
        if (
            self.cuda_reserved_bytes < self.cuda_allocated_bytes
            or self.cuda_peak_allocated_bytes < self.cuda_allocated_bytes
            or self.cuda_peak_reserved_bytes < self.cuda_reserved_bytes
        ):
            raise ProductionObservationError(
                "CUDA current/peak memory observations do not reconcile"
            )


def _host_memory_bytes() -> tuple[int, int]:
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = (
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                )

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.argtypes = ()
            get_current_process.restype = wintypes.HANDLE
            get_memory = psapi.GetProcessMemoryInfo
            get_memory.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCountersEx),
                wintypes.DWORD,
            )
            get_memory.restype = wintypes.BOOL
            counters = _ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            if not get_memory(
                get_current_process(),
                ctypes.byref(counters),
                counters.cb,
            ):
                raise OSError(
                    ctypes.get_last_error(),
                    "GetProcessMemoryInfo failed",
                )
            return int(counters.WorkingSetSize), int(
                counters.PeakWorkingSetSize
            )
        except Exception as exc:
            raise ProductionObservationError(
                "Windows process RSS/HWM observation failed"
            ) from exc
    try:
        import resource

        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        resident_pages = int(
            Path("/proc/self/statm")
            .read_text(encoding="ascii")
            .split()[1]
        )
        rss = resident_pages * page_size
        raw_hwm = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        hwm = raw_hwm if sys.platform == "darwin" else raw_hwm * 1024
        return rss, hwm
    except Exception as exc:
        raise ProductionObservationError(
            "POSIX process RSS/HWM observation failed"
        ) from exc


def _cuda_memory_bytes() -> tuple[int, int, int, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            raise ProductionObservationError(
                "production CUDA memory observation is unavailable"
            )
        device = torch.cuda.current_device()
        return (
            int(torch.cuda.memory_allocated(device)),
            int(torch.cuda.memory_reserved(device)),
            int(torch.cuda.max_memory_allocated(device)),
            int(torch.cuda.max_memory_reserved(device)),
        )
    except ProductionObservationError:
        raise
    except Exception as exc:
        raise ProductionObservationError(
            "CUDA memory observation failed"
        ) from exc


def capture_memory_observation(
    *,
    host_provider: Callable[[], tuple[int, int]] = _host_memory_bytes,
    cuda_provider: Callable[[], tuple[int, int, int, int]] = (
        _cuda_memory_bytes
    ),
) -> MemoryObservation:
    """Capture actual process and CUDA counters through injectable providers."""

    if not callable(host_provider) or not callable(cuda_provider):
        raise ProductionObservationError(
            "memory observation providers must be callable"
        )
    try:
        host = host_provider()
        cuda = cuda_provider()
    except ProductionObservationError:
        raise
    except Exception as exc:
        raise ProductionObservationError(
            "memory observation provider failed"
        ) from exc
    if (
        type(host) is not tuple
        or len(host) != 2
        or type(cuda) is not tuple
        or len(cuda) != 4
    ):
        raise ProductionObservationError(
            "memory observation providers returned malformed counters"
        )
    return MemoryObservation(
        process_rss_bytes=host[0],
        process_hwm_bytes=host[1],
        cuda_allocated_bytes=cuda[0],
        cuda_reserved_bytes=cuda[1],
        cuda_peak_allocated_bytes=cuda[2],
        cuda_peak_reserved_bytes=cuda[3],
    )


@dataclass(frozen=True, slots=True)
class PhaseTimingObservation:
    """Disjoint monotonic durations recorded around real operations."""

    data_wait_ns: int
    forward_ns: int
    inference_ns: int
    backward_ns: int
    update_ns: int
    evaluation_ns: int
    checkpoint_ns: int
    wall_ns: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if (
            any(type(value) is not int or value < 0 for value in values)
            or sum(values[:-1]) > self.wall_ns
        ):
            raise ProductionObservationError(
                "phase timings must be disjoint nonnegative wall-clock durations"
            )


_R = TypeVar("_R")


class PhaseTimer:
    """Wrap engine events with synchronized monotonic measurements."""

    _CATEGORY = {
        "forward": "forward",
        "cross_entropy": "forward",
        "complete_elbo": "forward",
        "emission_only_ablation_non_elbo": "forward",
        "recognition_forward": "inference",
        "immutable_detached_snapshot": "inference",
        "backward": "backward",
        "recognition_backward": "backward",
        "model_backward": "backward",
        "recognition_adamw": "update",
        "model_adamw": "update",
    }

    def __init__(
        self,
        *,
        monotonic_ns: Callable[[], int] = time.perf_counter_ns,
        synchronize: Callable[[], None] = lambda: None,
    ) -> None:
        if not callable(monotonic_ns) or not callable(synchronize):
            raise ProductionObservationError(
                "phase timer providers must be callable"
            )
        self._monotonic_ns = monotonic_ns
        self._synchronize = synchronize
        self._totals = {
            "forward": 0,
            "inference": 0,
            "backward": 0,
            "update": 0,
        }
        self._active = False

    def run(
        self,
        event_name: str,
        operation: Callable[[], _R],
    ) -> _R:
        category = self._CATEGORY.get(event_name)
        if category is None:
            raise ProductionObservationError(
                f"unclassified production event: {event_name}"
            )
        if self._active:
            raise ProductionObservationError(
                "production timing events cannot be nested"
            )
        if not callable(operation):
            raise ProductionObservationError(
                "timed production operation must be callable"
            )
        self._active = True
        self._synchronize()
        started = self._monotonic_ns()
        try:
            return operation()
        finally:
            self._synchronize()
            ended = self._monotonic_ns()
            self._active = False
            if (
                type(started) is not int
                or type(ended) is not int
                or ended < started
            ):
                raise ProductionObservationError(
                    "production monotonic clock moved backward"
                )
            self._totals[category] += ended - started

    def observation(
        self,
        *,
        data_wait_ns: int,
        evaluation_ns: int,
        checkpoint_ns: int,
        wall_ns: int,
    ) -> PhaseTimingObservation:
        if self._active:
            raise ProductionObservationError(
                "cannot snapshot an active timing event"
            )
        return PhaseTimingObservation(
            data_wait_ns=data_wait_ns,
            forward_ns=self._totals["forward"],
            inference_ns=self._totals["inference"],
            backward_ns=self._totals["backward"],
            update_ns=self._totals["update"],
            evaluation_ns=evaluation_ns,
            checkpoint_ns=checkpoint_ns,
            wall_ns=wall_ns,
        )


__all__ = [
    "MemoryObservation",
    "NumericalObservation",
    "PhaseTimer",
    "PhaseTimingObservation",
    "ProductionObservationError",
    "ProjectedMetric",
    "SourceObservation",
    "capture_memory_observation",
    "project_objective_metrics",
]
