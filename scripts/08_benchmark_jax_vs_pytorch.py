"""Same-T4 end-to-end benchmark of V1 NumPy/PyTorch and V2 JAX/PyTorch."""

# ruff: noqa: E402

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

LOGGER = logging.getLogger(__name__)

import jax
import jax.numpy as jnp

from hybrid_flood.benchmark.gpu_monitor import NvidiaMonitor
from hybrid_flood.comparison.inference import (
    dataset_for_forecast,
    predict_device_features,
    v1_features_to_torch,
)
from hybrid_flood.comparison.v2_staging import corrections_to_jax, v2_features_to_torch
from hybrid_flood.jax_solver.boundary_conditions import all_reflective
from hybrid_flood.jax_solver.numerics import integrate_to_outputs
from hybrid_flood.jax_solver.shallow_water_2d import ShallowWaterSolver
from hybrid_flood.ml.dataset import load_residual_dataset
from hybrid_flood.ml.evaluate import compute_test_metrics
from hybrid_flood.ml.torch_model import PyTorchResidualUNet
from hybrid_flood.ml.torch_train import load_checkpoint
from hybrid_flood.numpy_solver.shallow_water_2d import NumpyShallowWaterSolver
from hybrid_flood.structured_io import load_rainfall_series, load_structured_grid
from hybrid_flood.viz.static_figures import plot_v1_v2_benchmark


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _torch_model(cfg: DictConfig, inputs: int, device: torch.device) -> PyTorchResidualUNet:
    architecture = cfg.model.architecture
    model = PyTorchResidualUNet(
        input_channels=inputs,
        depth=int(architecture.depth),
        channels=tuple(int(value) for value in architecture.channels),
        activation=str(architecture.activation),
        kernel_size=int(architecture.kernel_size),
        output_channels=int(architecture.output_channels),
    ).to(device)
    load_checkpoint(
        model, _path(Path(get_original_cwd()), cfg.model.outputs.checkpoint), device=device
    )
    return model


def _summaries(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for version, group in frame.groupby("version"):
        result[str(version)] = {
            column: float(group[column].mean())
            for column in (
                "physics_execution_s",
                "feature_execution_s",
                "ai_inference_s",
                "postprocess_execution_s",
                "end_to_end_steady_s",
            )
        }
        result[str(version)]["end_to_end_std_s"] = float(group["end_to_end_steady_s"].std(ddof=1))
    return result


def _validate_jax_completion(result, expected_outputs: int, target_time_s: float) -> None:
    """Fail the benchmark if an optimized scan stops before the fixed horizon."""
    outputs, final_time, _ = jax.device_get(
        (result.outputs_written, result.final_time_s, result.steps_used)
    )
    if int(outputs) != expected_outputs or not np.isclose(float(final_time), target_time_s):
        raise RuntimeError(
            "V2 did not complete the configured forecast horizon: "
            f"outputs={int(outputs)}/{expected_outputs}, final_time={float(final_time):.6g}s."
        )


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Benchmark both versions repeatedly and reject non-T4 publication runs."""
    root = Path(get_original_cwd()).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    jax_gpus = [candidate for candidate in jax.devices() if candidate.platform == "gpu"]
    allow_cpu = bool(cfg.comparison.hardware.allow_cpu_smoke)
    if (device.type != "cuda" or not jax_gpus) and not allow_cpu:
        raise RuntimeError(
            "V1/V2 publication benchmarking requires one GPU visible to both frameworks."
        )
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    if bool(cfg.comparison.hardware.require_t4) and "T4" not in device_name.upper():
        raise RuntimeError(f"Publication benchmark requires NVIDIA T4, found {device_name!r}.")
    common = cfg.comparison.common
    grid = load_structured_grid(
        _path(root, common.inputs.dem),
        _path(root, common.inputs.roughness),
        _path(root, common.inputs.domain),
        resolution_m=float(common.resolution_m),
    )
    rain_times, rain_rates, default_rate = load_rainfall_series(
        _path(root, common.inputs.rainfall), scenario=str(common.inputs.rainfall_scenario)
    )
    numpy_solver = NumpyShallowWaterSolver(
        grid.bed,
        grid.manning_n,
        grid.domain_mask,
        resolution_m=grid.resolution_m,
        rainfall_times_s=rain_times,
        rainfall_rates_m_s=rain_rates,
        default_rainfall_m_s=default_rate,
        gravity_m_s2=float(common.gravity_m_s2),
        cfl=float(common.cfl),
        dry_tolerance_m=float(common.dry_tolerance_m),
        max_dt_s=float(common.max_dt_s),
    )
    jax_solver = ShallowWaterSolver(
        grid,
        rainfall_times_s=rain_times,
        rainfall_rates_m_s=rain_rates,
        default_rainfall_m_s=default_rate,
        boundaries=all_reflective(),
        gravity_m_s2=float(common.gravity_m_s2),
        cfl=float(common.cfl),
        dry_tolerance_m=float(common.dry_tolerance_m),
        max_dt_s=float(common.max_dt_s),
    )
    initial_kwargs = OmegaConf.to_container(common.initial_condition)
    numpy_initial = numpy_solver.initial_state(**initial_kwargs)
    jax_initial = jax_solver.initial_state(**initial_kwargs)
    times_host = np.arange(
        0.0,
        float(common.duration_s) + 0.5 * float(common.output_interval_s),
        float(common.output_interval_s),
        dtype=np.float32,
    )
    times_jax = jnp.asarray(times_host)
    start = perf_counter()
    jax_executable = integrate_to_outputs.lower(
        jax_initial, jax_solver.params, times_jax, max_steps=int(common.max_steps)
    ).compile()
    physics_compile_s = perf_counter() - start

    template = load_residual_dataset(_path(root, cfg.model.dataset.output))
    v1_dataset = dataset_for_forecast(
        template, _path(root, cfg.comparison.v1.output), _path(root, cfg.model.inputs.anuga)
    )
    v2_dataset = dataset_for_forecast(
        template, _path(root, cfg.comparison.v2.output), _path(root, cfg.model.inputs.anuga)
    )
    common_mask = v1_dataset.domain_mask & v2_dataset.domain_mask
    excluded_cells = int(template.domain_mask.sum() - common_mask.sum())
    v1_dataset = replace(
        v1_dataset,
        domain_mask=common_mask,
        loss_mask=v1_dataset.loss_mask & common_mask,
    )
    v2_dataset = replace(
        v2_dataset,
        domain_mask=common_mask,
        loss_mask=v2_dataset.loss_mask & common_mask,
    )
    LOGGER.info(
        "Benchmarking on %d common finite cells (%d excluded)",
        int(common_mask.sum()),
        excluded_cells,
    )
    model = _torch_model(cfg, template.input_channels, device)
    patch_size = int(cfg.model.training.patch_size)
    batch_size = int(cfg.model.training.batch_size)
    warmups = int(cfg.comparison.benchmark.warmup_runs)
    repetitions = int(cfg.comparison.benchmark.measured_runs)

    # Compile V2 feature/postprocessing once and record their first-call costs.
    v2_features, feature_compile_s, _ = v2_features_to_torch(v2_dataset)
    warm_prediction, _ = predict_device_features(
        model, v2_features, patch_size=patch_size, batch_size=batch_size
    )
    _, post_compile_s, _ = corrections_to_jax(v2_dataset.raw_depth_t_plus_1, warm_prediction)
    del warm_prediction, v2_features
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for warmup in range(warmups):
        LOGGER.info("Solver warmup %d/%d", warmup + 1, warmups)
        numpy_solver.run(numpy_initial, times_host, max_steps=int(common.max_steps))
        result = jax_executable(jax_initial, jax_solver.params, times_jax)
        jax.block_until_ready(result.states.h)
        _validate_jax_completion(result, len(times_host), float(times_host[-1]))
    if warmups:
        del result
    rows: list[dict[str, float | int | str]] = []
    monitors: dict[str, dict[str, float | int]] = {}
    last_predictions: dict[str, np.ndarray] = {}
    for version in ("V1", "V2"):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with NvidiaMonitor(float(cfg.comparison.benchmark.monitor_interval_s)) as monitor:
            for repetition in range(repetitions):
                LOGGER.info("%s measured repetition %d/%d", version, repetition + 1, repetitions)
                if version == "V1":
                    start = perf_counter()
                    numpy_solver.run(numpy_initial, times_host, max_steps=int(common.max_steps))
                    physics_s = perf_counter() - start
                    features, feature_s = v1_features_to_torch(v1_dataset, device)
                    prediction, ai_s = predict_device_features(
                        model, features, patch_size=patch_size, batch_size=batch_size
                    )
                    start = perf_counter()
                    prediction_host = prediction.permute(0, 2, 3, 1).cpu().numpy()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    post_s = perf_counter() - start
                else:
                    start = perf_counter()
                    result = jax_executable(jax_initial, jax_solver.params, times_jax)
                    jax.block_until_ready(result.states.h)
                    _validate_jax_completion(result, len(times_host), float(times_host[-1]))
                    physics_s = perf_counter() - start
                    features, _, feature_s = v2_features_to_torch(v2_dataset)
                    prediction, ai_s = predict_device_features(
                        model, features, patch_size=patch_size, batch_size=batch_size
                    )
                    _, _, post_s = corrections_to_jax(v2_dataset.raw_depth_t_plus_1, prediction)
                    prediction_host = prediction.permute(0, 2, 3, 1).cpu().numpy()
                last_predictions[version] = prediction_host
                total = physics_s + feature_s + ai_s + post_s
                rows.append(
                    {
                        "version": version,
                        "repetition": repetition,
                        "device": device_name,
                        "physics_execution_s": physics_s,
                        "feature_execution_s": feature_s,
                        "ai_inference_s": ai_s,
                        "postprocess_execution_s": post_s,
                        "end_to_end_steady_s": total,
                    }
                )
                del features, prediction
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        monitors[version] = monitor.summary()
        monitors[version]["torch_peak_memory_mib"] = (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else float("nan")
        )
    frame = pd.DataFrame(rows)
    summaries = _summaries(frame)
    speedup = summaries["V1"]["end_to_end_steady_s"] / summaries["V2"]["end_to_end_steady_s"]
    cold_v2 = (
        summaries["V2"]["end_to_end_steady_s"]
        + physics_compile_s
        + feature_compile_s
        + post_compile_s
    )
    cold_speedup = summaries["V1"]["end_to_end_steady_s"] / cold_v2
    metrics: dict[str, dict[str, float | int]] = {}
    threshold = float(cfg.comparison.accuracy.flood_threshold_m)
    for version, dataset in (("V1", v1_dataset), ("V2", v2_dataset)):
        indices = dataset.test_indices
        metrics[version] = compute_test_metrics(
            last_predictions[version][indices],
            dataset.target_residual[indices],
            dataset.raw_depth_t_plus_1[indices],
            dataset.domain_mask,
            flood_threshold_m=threshold,
        )
    stage_columns = {
        "physics": "physics_execution_s",
        "feature_assembly": "feature_execution_s",
        "ai_inference": "ai_inference_s",
        "postprocess": "postprocess_execution_s",
    }
    v1_bottlenecks = sorted(
        (
            {
                "stage": stage,
                "mean_seconds": summaries["V1"][column],
                "fraction_percent": 100.0
                * summaries["V1"][column]
                / summaries["V1"]["end_to_end_steady_s"],
            }
            for stage, column in stage_columns.items()
        ),
        key=lambda record: record["mean_seconds"],
        reverse=True,
    )
    csv_path = _path(root, cfg.comparison.benchmark.results_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    figure_paths = plot_v1_v2_benchmark(frame, _path(root, cfg.comparison.benchmark.figure_base))
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "measured_t4" if device.type == "cuda" else "cpu_smoke_only",
        "hardware": {"torch_device": device_name, "jax_devices": [str(d) for d in jax.devices()]},
        "configuration": OmegaConf.to_container(cfg.comparison, resolve=True),
        "methodology": {
            "same_anuga_reference": True,
            "same_pytorch_checkpoint": True,
            "same_inputs_and_split": True,
            "jax_compilation_excluded_from_steady_state": True,
            "speedup_definition": "mean V1 steady runtime / mean V2 steady runtime",
            "jax_torch_boundaries": 2,
        },
        "evaluation_domain": {
            "common_finite_cell_count": int(common_mask.sum()),
            "cells_excluded_for_nonfinite_v2_support": excluded_cells,
            "policy": "intersection of finite ANUGA, V1, and V2 support",
        },
        "jax_runtime_probe": JAX_RUNTIME,
        "jax_compilation_seconds": {
            "physics": physics_compile_s,
            "feature_assembly": feature_compile_s,
            "postprocess": post_compile_s,
            "total": physics_compile_s + feature_compile_s + post_compile_s,
        },
        "summary": summaries,
        "speedup_v1_over_v2": speedup,
        "cold_start_speedup_v1_over_v2": cold_speedup,
        "v1_bottlenecks": v1_bottlenecks,
        "hardware_samples": monitors,
        "test_metrics": metrics,
        "artifacts": {"csv": str(csv_path), "figures": [str(path) for path in figure_paths]},
    }
    metadata_path = _path(root, cfg.comparison.benchmark.metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    report_directory = _path(root, cfg.comparison.benchmark.report_directory)
    report_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, report_directory / csv_path.name)
    for path in figure_paths:
        shutil.copy2(path, report_directory / path.name)
    LOGGER.info("V1/V2 T4 benchmark complete: speedup %.6gx", speedup)


if __name__ == "__main__":
    main()
