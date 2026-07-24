"""JAX runtime selection without importing JAX or initializing its plugins.

The project carries CUDA-enabled JAX wheels for HPC execution.  On a workstation
without an NVIDIA driver, importing JAX otherwise emits a long CUDA plugin
traceback before falling back to CPU.  This module probes the NVIDIA driver
first and selects the CPU backend cleanly when CUDA cannot possibly initialize.
"""

from __future__ import annotations

import ctypes
import os
from ctypes.util import find_library
from typing import Any


def probe_nvidia_cuda_driver() -> dict[str, Any]:
    """Return whether the NVIDIA CUDA driver exists and accepts ``cuInit(0)``."""
    discovered = find_library("cuda")
    candidates = [name for name in (discovered, "libcuda.so.1", "libcuda.so") if name]
    for library_name in dict.fromkeys(candidates):
        try:
            driver = ctypes.CDLL(library_name)
        except OSError:
            continue

        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        result = int(driver.cuInit(0))
        if result == 0:
            return {
                "available": True,
                "library": library_name,
                "cu_init_result": result,
                "reason": "NVIDIA CUDA driver initialized successfully.",
            }
        return {
            "available": False,
            "library": library_name,
            "cu_init_result": result,
            "reason": f"NVIDIA CUDA driver cuInit(0) returned error code {result}.",
        }

    return {
        "available": False,
        "library": None,
        "cu_init_result": None,
        "reason": "NVIDIA CUDA driver library was not found.",
    }


def configure_jax_runtime() -> dict[str, Any]:
    """Select CPU before JAX import when no usable NVIDIA driver is present.

    An explicit ``JAX_PLATFORMS`` value always wins.  The returned dictionary is
    JSON-serializable and should be included in run metadata.
    """
    explicit_platforms = os.environ.get("JAX_PLATFORMS")
    probe = probe_nvidia_cuda_driver()
    forced_cpu = False
    if explicit_platforms is None and not probe["available"]:
        os.environ["JAX_PLATFORMS"] = "cpu"
        forced_cpu = True

    return {
        "cuda_driver": probe,
        "explicit_jax_platforms": explicit_platforms,
        "effective_jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "automatic_cpu_fallback": forced_cpu,
    }
