"""Repeated, synchronized benchmarks for matched JAX and PyTorch U-Nets.

Input creation, model initialization, and layout conversion are excluded from
timed regions. JAX uses NHWC and PyTorch uses NCHW, their respective optimized
convolution layouts. Every measured GPU interval is explicitly synchronized.
The forward+backward operation includes AdamW parameter updates in both
frameworks and uses the same scalar mean-square objective.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import torch

from hybrid_flood.benchmark.pytorch_residual_net import (
    TorchResidualUNet,
    count_torch_parameters,
)
from hybrid_flood.ml.residual_net import ResidualUNet


@dataclass(frozen=True)
class BenchmarkRecord:
    """One repeated timing configuration."""

    framework: str
    framework_version: str
    device_type: str
    device_name: str
    batch_size: int
    input_height: int
    input_width: int
    input_channels: int
    operation: str
    compiled: bool
    compile_or_first_iteration_ms: float
    mean_ms: float
    std_ms: float
    minimum_ms: float
    maximum_ms: float
    warmup_iterations: int
    measured_iterations: int
    peak_memory_mb: float
    memory_measurement: str
    parameter_count: int


def count_jax_parameters(params: Any) -> int:
    """Count scalar parameters in a Flax parameter pytree."""
    return int(sum(np.asarray(value).size for value in jax.tree.leaves(params)))


def _timing_summary(samples_seconds: list[float]) -> dict[str, float]:
    samples_ms = np.asarray(samples_seconds, dtype=np.float64) * 1.0e3
    return {
        "mean_ms": float(samples_ms.mean()),
        "std_ms": float(samples_ms.std(ddof=1)) if len(samples_ms) > 1 else 0.0,
        "minimum_ms": float(samples_ms.min()),
        "maximum_ms": float(samples_ms.max()),
    }


def _jax_memory(device: jax.Device) -> tuple[float, str]:
    statistics = device.memory_stats()
    if not statistics:
        return float("nan"), "unavailable on this backend"
    for key in ("peak_bytes_in_use", "peak_bytes_in_use_sum"):
        if key in statistics:
            return float(statistics[key]) / 1024.0**2, f"JAX allocator {key}"
    if "bytes_in_use" in statistics:
        return float(statistics["bytes_in_use"]) / 1024.0**2, "JAX allocator bytes_in_use"
    return float("nan"), "JAX allocator did not expose memory bytes"


def _measure_jax(
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> tuple[float, list[float], Any]:
    start = perf_counter()
    result = function(*arguments)
    jax.block_until_ready(result)
    first_seconds = perf_counter() - start
    for _ in range(warmup_iterations):
        result = function(*arguments)
        jax.block_until_ready(result)
    samples: list[float] = []
    for _ in range(measured_iterations):
        start = perf_counter()
        result = function(*arguments)
        jax.block_until_ready(result)
        samples.append(perf_counter() - start)
    return first_seconds, samples, result


def benchmark_jax_model(
    model: ResidualUNet,
    *,
    batch_size: int,
    input_shape: tuple[int, int, int],
    warmup_iterations: int,
    measured_iterations: int,
    seed: int = 42,
) -> list[BenchmarkRecord]:
    """Benchmark JIT forward and AdamW training steps on JAX's default device."""
    height, width, channels = input_shape
    key = jax.random.PRNGKey(seed)
    logical_inputs = np.random.default_rng(seed).standard_normal(
        (batch_size, height, width, channels),
        dtype=np.float32,
    )
    inputs = jnp.asarray(logical_inputs)
    params = model.init(key, inputs)["params"]
    parameter_count = count_jax_parameters(params)
    device = jax.devices()[0]

    @jax.jit
    def forward(model_params, batch):
        return model.apply({"params": model_params}, batch)

    compile_seconds, samples, _ = _measure_jax(
        forward,
        (params, inputs),
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    peak_memory, memory_method = _jax_memory(device)
    records = [
        BenchmarkRecord(
            framework="JAX",
            framework_version=jax.__version__,
            device_type=device.platform,
            device_name=str(device),
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            input_channels=channels,
            operation="forward",
            compiled=True,
            compile_or_first_iteration_ms=compile_seconds * 1.0e3,
            **_timing_summary(samples),
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            peak_memory_mb=peak_memory,
            memory_measurement=memory_method,
            parameter_count=parameter_count,
        )
    ]

    optimizer = optax.adamw(1.0e-3)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def training_step(model_params, state, batch):
        def objective(candidate):
            prediction = model.apply({"params": candidate}, batch)
            return jnp.mean(jnp.square(prediction - 1.0))

        loss, gradients = jax.value_and_grad(objective)(model_params)
        updates, next_state = optimizer.update(gradients, state, model_params)
        next_params = optax.apply_updates(model_params, updates)
        return next_params, next_state, loss

    start = perf_counter()
    params, optimizer_state, loss = training_step(params, optimizer_state, inputs)
    jax.block_until_ready(loss)
    first_seconds = perf_counter() - start
    for _ in range(warmup_iterations):
        params, optimizer_state, loss = training_step(params, optimizer_state, inputs)
        jax.block_until_ready(loss)
    samples = []
    for _ in range(measured_iterations):
        start = perf_counter()
        params, optimizer_state, loss = training_step(params, optimizer_state, inputs)
        jax.block_until_ready(loss)
        samples.append(perf_counter() - start)
    peak_memory, memory_method = _jax_memory(device)
    records.append(
        BenchmarkRecord(
            framework="JAX",
            framework_version=jax.__version__,
            device_type=device.platform,
            device_name=str(device),
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            input_channels=channels,
            operation="forward_backward_adamw",
            compiled=True,
            compile_or_first_iteration_ms=first_seconds * 1.0e3,
            **_timing_summary(samples),
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            peak_memory_mb=peak_memory,
            memory_measurement=memory_method,
            parameter_count=parameter_count,
        )
    )
    return records


def _torch_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _torch_peak_memory(device: torch.device) -> tuple[float, str]:
    if device.type != "cuda":
        return float("nan"), "unavailable on CPU"
    return (
        float(torch.cuda.max_memory_allocated(device)) / 1024.0**2,
        "torch.cuda.max_memory_allocated after warmup reset",
    )


def _torch_compile(model: torch.nn.Module, enabled: bool, mode: str) -> torch.nn.Module:
    return torch.compile(model, mode=mode) if enabled else model


def benchmark_torch_model(
    model_factory: Callable[[], TorchResidualUNet],
    *,
    batch_size: int,
    input_shape: tuple[int, int, int],
    warmup_iterations: int,
    measured_iterations: int,
    device: torch.device,
    compile_model: bool,
    compile_mode: str = "default",
    seed: int = 42,
) -> list[BenchmarkRecord]:
    """Benchmark PyTorch forward and AdamW steps with CUDA synchronization."""
    height, width, channels = input_shape
    torch.manual_seed(seed)
    model = model_factory().to(device)
    parameter_count = count_torch_parameters(model)
    logical_inputs = np.random.default_rng(seed).standard_normal(
        (batch_size, height, width, channels),
        dtype=np.float32,
    )
    inputs = torch.from_numpy(np.ascontiguousarray(logical_inputs.transpose(0, 3, 1, 2))).to(device)
    compiled_model = _torch_compile(model, compile_model, compile_mode)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        start = perf_counter()
        output = compiled_model(inputs)
        _torch_synchronize(device)
        first_seconds = perf_counter() - start
        for _ in range(warmup_iterations):
            output = compiled_model(inputs)
            _torch_synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    with torch.no_grad():
        for _ in range(measured_iterations):
            start = perf_counter()
            output = compiled_model(inputs)
            _torch_synchronize(device)
            samples.append(perf_counter() - start)
    peak_memory, memory_method = _torch_peak_memory(device)
    records = [
        BenchmarkRecord(
            framework="PyTorch",
            framework_version=torch.__version__,
            device_type=device.type,
            device_name=(torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            input_channels=channels,
            operation="forward",
            compiled=compile_model,
            compile_or_first_iteration_ms=first_seconds * 1.0e3,
            **_timing_summary(samples),
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            peak_memory_mb=peak_memory,
            memory_measurement=memory_method,
            parameter_count=parameter_count,
        )
    ]

    del output, compiled_model, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = model_factory().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    compiled_model = _torch_compile(model, compile_model, compile_mode)

    def training_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        prediction = compiled_model(inputs)
        loss = (prediction - 1.0).square().mean()
        loss.backward()
        optimizer.step()
        return loss

    start = perf_counter()
    training_step()
    _torch_synchronize(device)
    first_seconds = perf_counter() - start
    for _ in range(warmup_iterations):
        training_step()
        _torch_synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(measured_iterations):
        start = perf_counter()
        training_step()
        _torch_synchronize(device)
        samples.append(perf_counter() - start)
    peak_memory, memory_method = _torch_peak_memory(device)
    records.append(
        BenchmarkRecord(
            framework="PyTorch",
            framework_version=torch.__version__,
            device_type=device.type,
            device_name=(torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            input_channels=channels,
            operation="forward_backward_adamw",
            compiled=compile_model,
            compile_or_first_iteration_ms=first_seconds * 1.0e3,
            **_timing_summary(samples),
            warmup_iterations=warmup_iterations,
            measured_iterations=measured_iterations,
            peak_memory_mb=peak_memory,
            memory_measurement=memory_method,
            parameter_count=parameter_count,
        )
    )
    return records


def benchmark_frameworks(
    flax_model: ResidualUNet,
    torch_model_factory: Callable[[], TorchResidualUNet],
    *,
    batch_sizes: list[int],
    input_shape: tuple[int, int, int],
    warmup_iterations: int,
    measured_iterations: int,
    require_gpu: bool = True,
    allow_cpu: bool = False,
    torch_compile_enabled: bool = True,
    torch_compile_mode: str = "default",
    seed: int = 42,
) -> pd.DataFrame:
    """Run both frameworks on one device class and return tidy measurements."""
    if not batch_sizes or any(batch < 1 for batch in batch_sizes):
        raise ValueError("Batch sizes must be a non-empty list of positive integers.")
    if warmup_iterations < 0 or measured_iterations < 2:
        raise ValueError("Use non-negative warmup and at least two measured iterations.")
    jax_gpu = next((device for device in jax.devices() if device.platform == "gpu"), None)
    torch_gpu = torch.cuda.is_available()
    gpu_ready = jax_gpu is not None and torch_gpu
    if require_gpu and not gpu_ready:
        raise RuntimeError(
            "A fair GPU benchmark requires both JAX and PyTorch to see the same NVIDIA GPU; "
            f"JAX devices={jax.devices()}, torch.cuda.is_available()={torch_gpu}."
        )
    if not gpu_ready and not allow_cpu:
        raise RuntimeError(
            "GPU unavailable and CPU fallback was not explicitly allowed. "
            "Set allow_cpu=true only for a labelled smoke test, not report results."
        )
    torch_device = torch.device("cuda:0" if gpu_ready else "cpu")

    records: list[BenchmarkRecord] = []
    for batch_size in batch_sizes:
        records.extend(
            benchmark_jax_model(
                flax_model,
                batch_size=batch_size,
                input_shape=input_shape,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
                seed=seed,
            )
        )
        records.extend(
            benchmark_torch_model(
                torch_model_factory,
                batch_size=batch_size,
                input_shape=input_shape,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
                device=torch_device,
                compile_model=torch_compile_enabled,
                compile_mode=torch_compile_mode,
                seed=seed,
            )
        )
        gc.collect()
    frame = pd.DataFrame(asdict(record) for record in records)
    if frame.groupby("framework")["parameter_count"].first().nunique() != 1:
        raise RuntimeError("JAX and PyTorch parameter counts differ; benchmark is not fair.")
    return frame.sort_values(["operation", "batch_size", "framework"]).reset_index(drop=True)


def benchmark_frameworks_isolated(
    *,
    architecture: dict[str, Any],
    batch_sizes: list[int],
    input_shape: tuple[int, int, int],
    warmup_iterations: int,
    measured_iterations: int,
    require_gpu: bool = True,
    allow_cpu: bool = False,
    torch_compile_enabled: bool = True,
    torch_compile_mode: str = "default",
    seed: int = 42,
) -> pd.DataFrame:
    """Benchmark each batch in a fresh process for resettable memory peaks.

    JAX's CUDA allocator reports a process-level peak and exposes no public
    reset equivalent to ``torch.cuda.reset_peak_memory_stats``. A fresh worker
    per batch makes the JAX peak attributable to that batch rather than a
    previous, larger configuration. Process startup is outside all timings.
    """
    if not batch_sizes or any(int(batch) < 1 for batch in batch_sizes):
        raise ValueError("Batch sizes must be a non-empty list of positive integers.")
    records: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="hybrid_flood_benchmark_") as temporary:
        temporary_directory = Path(temporary)
        for position, batch_size in enumerate(sorted(set(int(value) for value in batch_sizes))):
            payload = {
                "architecture": architecture,
                "batch_size": batch_size,
                "input_shape": list(input_shape),
                "warmup_iterations": warmup_iterations,
                "measured_iterations": measured_iterations,
                "require_gpu": require_gpu,
                "allow_cpu": allow_cpu,
                "torch_compile_enabled": torch_compile_enabled,
                "torch_compile_mode": torch_compile_mode,
                "seed": seed,
            }
            payload_path = temporary_directory / f"payload_{position}.json"
            result_path = temporary_directory / f"result_{position}.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hybrid_flood.benchmark.worker",
                    "--payload",
                    str(payload_path),
                    "--output",
                    str(result_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Isolated benchmark worker failed for batch {batch_size}:\n"
                    f"{completed.stdout}\n{completed.stderr}"
                )
            records.append(pd.read_json(result_path, orient="records"))
    frame = pd.concat(records, ignore_index=True)
    jax_rows = frame["framework"] == "JAX"
    frame.loc[jax_rows, "memory_measurement"] = (
        frame.loc[jax_rows, "memory_measurement"].astype(str) + "; fresh process per batch size"
    )
    if frame.groupby("framework")["parameter_count"].first().nunique() != 1:
        raise RuntimeError("JAX and PyTorch parameter counts differ; benchmark is not fair.")
    return frame.sort_values(["operation", "batch_size", "framework"]).reset_index(drop=True)


def save_benchmark_csv(results: pd.DataFrame, path: str | Path) -> Path:
    """Save tidy benchmark measurements."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)
    return destination


def format_latex_table(results: pd.DataFrame, path: str | Path) -> Path:
    """Write a compact report table with mean ± standard deviation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in results.itertuples(index=False):
        memory = "--" if not np.isfinite(record.peak_memory_mb) else f"{record.peak_memory_mb:.1f}"
        operation = record.operation.replace("_", r"\_")
        rows.append(
            f"{record.framework} & {operation} & "
            f"{record.batch_size} & {record.mean_ms:.3f} $\\pm$ {record.std_ms:.3f} & "
            f"{record.compile_or_first_iteration_ms:.1f} & {memory} \\\\"
        )
    contents = "\n".join(
        (
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Framework & Operation & Batch & Time (ms) & First (ms) & Peak MiB \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        )
    )
    destination.write_text(contents, encoding="utf-8")
    return destination


def save_benchmark_metadata(metadata: dict[str, Any], path: str | Path) -> Path:
    """Save device and methodology metadata alongside timing results."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return destination
