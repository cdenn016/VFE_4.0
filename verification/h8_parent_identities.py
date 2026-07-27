"""Parent-side collection of the frozen H8 runtime identity inventory."""

from __future__ import annotations

import contextlib
import ctypes
import io
import os
import platform
import sys
from collections.abc import Mapping
from typing import Any

from verification.h8_budget import make_h8_identity_record
from verification.h8_wire import H8_THREAD_ENVIRONMENT_ITEMS


def _hardware_payload() -> dict[str, object]:
    payload = {
        "platform": platform.platform(),
        "release": platform.release(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    if payload["cpu_count"] is None:
        raise ValueError("hardware CPU count is unavailable")
    return payload


def _affinity_payload() -> dict[str, object]:
    if hasattr(os, "sched_getaffinity"):
        return {
            "adapter": "os.sched_getaffinity",
            "cpus": sorted(os.sched_getaffinity(0)),
        }
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_process = kernel32.GetCurrentProcess
        get_process.restype = ctypes.c_void_p
        get_affinity = kernel32.GetProcessAffinityMask
        get_affinity.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        process = get_process()
        if not process:
            raise ValueError("GetCurrentProcess returned a null affinity handle")
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not get_affinity(
            process,
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        ):
            code = ctypes.get_last_error()
            raise ValueError(
                f"GetProcessAffinityMask failed with error {code}"
            )
        return {
            "adapter": "GetProcessAffinityMask",
            "process_mask": process_mask.value,
            "system_mask": system_mask.value,
        }
    raise ValueError("process affinity API is unavailable")


def _blas_payload(torch: Any, np: Any) -> dict[str, object]:
    numpy_buffer = io.StringIO()
    with contextlib.redirect_stdout(numpy_buffer):
        np.show_config()
    return {
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "torch_config": str(torch.__config__.show()),
        "numpy_config": numpy_buffer.getvalue(),
    }


def collect_h8_runtime_identities() -> Mapping[str, object]:
    """Collect the exact hardware, affinity, thread, and BLAS identities."""

    import numpy as np
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        if int(torch.get_num_interop_threads()) != 1:
            raise ValueError(
                "H8 parent interop threads cannot be frozen to one"
            ) from error
    if (
        int(torch.get_num_threads()) != 1
        or int(torch.get_num_interop_threads()) != 1
    ):
        raise ValueError("H8 parent torch threads are not frozen to one")
    thread_environment = dict(H8_THREAD_ENVIRONMENT_ITEMS)
    return {
        name: make_h8_identity_record(name, payload)
        for name, payload in (
            ("hardware", _hardware_payload()),
            ("affinity", _affinity_payload()),
            (
                "thread",
                {
                    "environment": thread_environment,
                    "torch_num_threads": int(torch.get_num_threads()),
                    "torch_num_interop_threads": int(
                        torch.get_num_interop_threads()
                    ),
                },
            ),
            ("blas", _blas_payload(torch, np)),
        )
    }


__all__ = ["collect_h8_runtime_identities"]
