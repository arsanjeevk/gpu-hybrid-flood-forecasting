"""Low-overhead NVIDIA utilization sampling for end-to-end experiments."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from time import sleep

import numpy as np


@dataclass
class NvidiaMonitor:
    """Poll one visible GPU without changing its state."""

    interval_s: float = 0.2
    utilization_percent: list[float] = field(default_factory=list)
    memory_mib: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def _sample(self) -> None:
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                try:
                    utilization, memory = completed.stdout.splitlines()[0].split(",")
                    self.utilization_percent.append(float(utilization.strip()))
                    self.memory_mib.append(float(memory.strip()))
                except ValueError:
                    pass
            sleep(self.interval_s)

    def __enter__(self) -> NvidiaMonitor:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.interval_s))

    def summary(self) -> dict[str, float | int]:
        """Return sample count, mean/peak utilization, and peak memory."""
        utilization = np.asarray(self.utilization_percent, dtype=np.float64)
        memory = np.asarray(self.memory_mib, dtype=np.float64)
        return {
            "sample_count": int(utilization.size),
            "gpu_utilization_mean_percent": (
                float(utilization.mean()) if utilization.size else float("nan")
            ),
            "gpu_utilization_peak_percent": (
                float(utilization.max()) if utilization.size else float("nan")
            ),
            "gpu_memory_peak_mib": float(memory.max()) if memory.size else float("nan"),
        }
