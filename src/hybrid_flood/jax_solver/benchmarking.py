"""Synchronized JAX step benchmarks that exclude compilation warmup."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

from hybrid_flood.jax_solver.numerics import SWEParams, SWEState, finite_volume_step


def benchmark_compiled_steps(
    state: SWEState,
    params: SWEParams,
    *,
    iterations: int = 100,
    dt_s: float = 1.0,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark repeated compiled steps on each available CPU/GPU backend."""
    if iterations < 1:
        raise ValueError("Benchmark iterations must be positive.")
    records: list[dict[str, Any]] = []
    available = jax.devices()
    for platform in ("cpu", "gpu"):
        matching = [device for device in available if device.platform == platform]
        if not matching:
            records.append(
                {
                    "platform": platform,
                    "available": False,
                    "reason": f"No JAX {platform.upper()} device is available.",
                }
            )
            continue
        device = matching[0]
        device_state = jax.device_put(state, device)
        device_params = jax.device_put(params, device)
        dt = jax.device_put(jnp.asarray(dt_s, state.h.dtype), device)

        def repeated_steps(initial_state):
            return jax.lax.fori_loop(
                0,
                iterations,
                lambda index, current: finite_volume_step(
                    current,
                    device_params,
                    dt,
                    index * dt,
                ),
                initial_state,
            )

        compiled = jax.jit(repeated_steps, device=device)
        jax.block_until_ready(compiled(device_state).h)  # JIT warmup excluded below.
        start = perf_counter()
        result = compiled(device_state)
        jax.block_until_ready(result.h)
        elapsed = perf_counter() - start
        records.append(
            {
                "platform": platform,
                "available": True,
                "device": str(device),
                "iterations": iterations,
                "total_seconds": elapsed,
                "seconds_per_step": elapsed / iterations,
                "steps_per_second": iterations / elapsed,
                "grid_shape": list(state.h.shape),
                "dtype": str(state.h.dtype),
                "includes_jit_compilation": False,
            }
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in available],
        "gpu_benchmark_available": any(
            record["platform"] == "gpu" and record["available"] for record in records
        ),
        "benchmarks": records,
    }
    if output_directory is not None:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / f"benchmark_{timestamp}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report["output_path"] = str(path)
    return report
