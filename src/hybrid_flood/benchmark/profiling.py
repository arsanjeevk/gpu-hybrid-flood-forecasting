"""Comparable Chrome-trace profiling wrappers for JAX and PyTorch."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import pandas as pd
import torch

from hybrid_flood.ml.residual_net import ResidualUNet


@dataclass(frozen=True)
class ProfileArtifacts:
    """Raw trace and aggregated operation table."""

    trace_path: Path
    summary_path: Path


def _find_jax_trace(directory: Path) -> Path:
    candidates = sorted(
        (
            *directory.rglob("*.trace.json.gz"),
            *directory.rglob("*.trace.json"),
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise FileNotFoundError(f"JAX profiler created no Chrome trace under {directory}.")
    return candidates[-1]


def summarize_chrome_trace(
    trace_path: str | Path,
    *,
    framework: str,
    output_path: str | Path,
    maximum_operations: int = 100,
) -> Path:
    """Aggregate complete Chrome-trace events into a common CSV schema."""
    source = Path(trace_path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    rows = [
        {
            "framework": framework,
            "operation": str(event.get("name", "unknown")),
            "category": str(event.get("cat", "")),
            "duration_us": float(event["dur"]),
        }
        for event in events
        if event.get("ph") == "X"
        and isinstance(event.get("dur"), (int, float))
        and event["dur"] >= 0
    ]
    if not rows:
        raise ValueError(f"Trace {source} contains no complete duration events.")
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["framework", "operation", "category"], as_index=False)
        .agg(
            calls=("duration_us", "size"),
            total_duration_us=("duration_us", "sum"),
            mean_duration_us=("duration_us", "mean"),
            std_duration_us=("duration_us", "std"),
        )
        .fillna({"std_duration_us": 0.0})
        .sort_values("total_duration_us", ascending=False)
        .head(maximum_operations)
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)
    return destination


def profile_jax_forward(
    model: ResidualUNet,
    params: Any,
    inputs: jax.Array,
    output_directory: str | Path,
    *,
    iterations: int = 10,
) -> ProfileArtifacts:
    """Capture a JAX/Perfetto trace for compiled model forward passes."""
    if iterations < 1:
        raise ValueError("Profiler iterations must be positive.")
    directory = Path(output_directory) / "jax_trace"
    directory.mkdir(parents=True, exist_ok=True)
    forward = jax.jit(lambda model_params, batch: model.apply({"params": model_params}, batch))
    jax.block_until_ready(forward(params, inputs))
    with jax.profiler.trace(str(directory), create_perfetto_link=False):
        for _ in range(iterations):
            jax.block_until_ready(forward(params, inputs))
    trace_path = _find_jax_trace(directory)
    summary_path = summarize_chrome_trace(
        trace_path,
        framework="JAX",
        output_path=Path(output_directory) / "jax_profile_operations.csv",
    )
    return ProfileArtifacts(trace_path, summary_path)


def profile_torch_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    output_directory: str | Path,
    *,
    iterations: int = 10,
) -> ProfileArtifacts:
    """Capture a PyTorch Chrome trace and common operation summary."""
    if iterations < 1:
        raise ValueError("Profiler iterations must be positive.")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = inputs.device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    model.eval()
    with torch.no_grad():
        model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for _ in range(iterations):
                model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                profiler.step()
    trace_path = directory / "pytorch_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    summary_path = summarize_chrome_trace(
        trace_path,
        framework="PyTorch",
        output_path=directory / "pytorch_profile_operations.csv",
    )
    return ProfileArtifacts(trace_path, summary_path)


def merge_profile_summaries(
    artifacts: list[ProfileArtifacts],
    output_path: str | Path,
) -> Path:
    """Concatenate framework operation summaries for report-table analysis."""
    if not artifacts:
        raise ValueError("At least one profile artifact is required.")
    combined = pd.concat(
        (pd.read_csv(artifact.summary_path) for artifact in artifacts),
        ignore_index=True,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return destination
