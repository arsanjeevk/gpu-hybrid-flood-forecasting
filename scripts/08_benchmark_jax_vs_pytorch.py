"""Run the architecture-matched JAX-versus-PyTorch benchmark."""

# ruff: noqa: E402

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

import jax
import jax.numpy as jnp
import torch

from hybrid_flood.benchmark.jax_vs_pytorch import (
    benchmark_frameworks_isolated,
    count_jax_parameters,
    format_latex_table,
    save_benchmark_csv,
    save_benchmark_metadata,
)
from hybrid_flood.benchmark.profiling import (
    merge_profile_summaries,
    profile_jax_forward,
    profile_torch_forward,
)
from hybrid_flood.benchmark.pytorch_residual_net import (
    TorchResidualUNet,
    count_torch_parameters,
)
from hybrid_flood.ml.residual_net import model_from_config
from hybrid_flood.viz.static_figures import plot_framework_benchmark

LOGGER = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.WARNING)


def _path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else root / path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig) -> None:
    """Benchmark matched models and export report-ready artifacts."""
    root = Path(get_original_cwd()).resolve()
    benchmark_cfg = cfg.benchmark
    model_cfg = cfg.model.architecture
    input_shape = (
        int(benchmark_cfg.input.height),
        int(benchmark_cfg.input.width),
        int(benchmark_cfg.input.channels),
    )
    if int(benchmark_cfg.input.channels) != 6:
        raise ValueError("The Phase 5 residual model has exactly six input channels.")

    flax_model = model_from_config(model_cfg)

    def torch_factory() -> TorchResidualUNet:
        return TorchResidualUNet(
            input_channels=input_shape[2],
            depth=int(model_cfg.depth),
            channels=tuple(int(value) for value in model_cfg.channels),
            activation=str(model_cfg.activation),
            kernel_size=int(model_cfg.kernel_size),
            output_channels=int(model_cfg.output_channels),
        )

    LOGGER.info(
        "Benchmarking batch sizes %s with %d warmups and %d measured iterations",
        list(benchmark_cfg.batch_sizes),
        benchmark_cfg.warmup_iterations,
        benchmark_cfg.measured_iterations,
    )
    results = benchmark_frameworks_isolated(
        architecture=OmegaConf.to_container(model_cfg, resolve=True),
        batch_sizes=[int(value) for value in benchmark_cfg.batch_sizes],
        input_shape=input_shape,
        warmup_iterations=int(benchmark_cfg.warmup_iterations),
        measured_iterations=int(benchmark_cfg.measured_iterations),
        require_gpu=bool(benchmark_cfg.require_gpu),
        allow_cpu=bool(benchmark_cfg.allow_cpu),
        torch_compile_enabled=bool(benchmark_cfg.torch.compile),
        torch_compile_mode=str(benchmark_cfg.torch.compile_mode),
        seed=int(cfg.project.seed),
    )

    output_directory = _path(root, benchmark_cfg.outputs.directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = save_benchmark_csv(
        results,
        output_directory / str(benchmark_cfg.outputs.results_csv),
    )
    latex_path = format_latex_table(
        results,
        output_directory / str(benchmark_cfg.outputs.latex_table),
    )
    figure_paths = plot_framework_benchmark(
        results,
        _path(root, benchmark_cfg.outputs.figure_base),
    )

    profile_outputs: list[str] = []
    if bool(benchmark_cfg.profiling.enabled):
        profile_batch = int(benchmark_cfg.profiling.batch_size)
        height, width, channels = input_shape
        key = jax.random.PRNGKey(int(cfg.project.seed))
        logical_inputs = np.random.default_rng(int(cfg.project.seed)).standard_normal(
            (profile_batch, height, width, channels),
            dtype=np.float32,
        )
        jax_inputs = jnp.asarray(logical_inputs)
        jax_params = flax_model.init(key, jax_inputs)["params"]
        torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_model = torch_factory().to(torch_device)
        if bool(benchmark_cfg.torch.compile):
            torch_model = torch.compile(
                torch_model,
                mode=str(benchmark_cfg.torch.compile_mode),
            )
        torch_inputs = torch.from_numpy(
            np.ascontiguousarray(logical_inputs.transpose(0, 3, 1, 2))
        ).to(torch_device)
        profile_directory = output_directory / str(benchmark_cfg.outputs.profile_directory)
        jax_profile = profile_jax_forward(
            flax_model,
            jax_params,
            jax_inputs,
            profile_directory,
            iterations=int(benchmark_cfg.profiling.iterations),
        )
        torch_profile = profile_torch_forward(
            torch_model,
            torch_inputs,
            profile_directory,
            iterations=int(benchmark_cfg.profiling.iterations),
        )
        profile_summary = merge_profile_summaries(
            [jax_profile, torch_profile],
            output_directory / str(benchmark_cfg.outputs.profile_summary_csv),
        )
        profile_outputs = [
            str(jax_profile.trace_path),
            str(jax_profile.summary_path),
            str(torch_profile.trace_path),
            str(torch_profile.summary_path),
            str(profile_summary),
        ]

    sample = jnp.zeros((1, *input_shape), dtype=jnp.float32)
    flax_params = flax_model.init(jax.random.PRNGKey(int(cfg.project.seed)), sample)["params"]
    torch_parameter_count = count_torch_parameters(torch_factory())
    jax_parameter_count = count_jax_parameters(flax_params)
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": OmegaConf.to_container(benchmark_cfg, resolve=True),
        "architecture": OmegaConf.to_container(model_cfg, resolve=True),
        "fairness": {
            "jax_parameter_count": jax_parameter_count,
            "pytorch_parameter_count": torch_parameter_count,
            "parameter_counts_equal": jax_parameter_count == torch_parameter_count,
            "input_semantics": "same float32 logical tensor; NHWC for JAX, NCHW for PyTorch",
            "timing": "synchronized wall clock; input creation and layout conversion excluded",
            "training_operation": "forward + MSE backward + AdamW update",
            "profiling_execution": "compiled steady-state forward passes in both frameworks",
        },
        "devices": {
            "jax": [str(device) for device in jax.devices()],
            "pytorch_cuda_available": torch.cuda.is_available(),
            "pytorch_cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "jax_runtime_probe": JAX_RUNTIME,
        },
        "artifacts": {
            "results_csv": str(csv_path),
            "latex_table": str(latex_path),
            "figures": [str(path) for path in figure_paths],
            "profiles": profile_outputs,
        },
    }
    metadata_path = save_benchmark_metadata(
        metadata,
        output_directory / str(benchmark_cfg.outputs.metadata),
    )

    report_directory = _path(root, benchmark_cfg.outputs.report_directory)
    report_directory.mkdir(parents=True, exist_ok=True)
    for artifact in (*figure_paths, csv_path, latex_path):
        shutil.copy2(artifact, report_directory / artifact.name)
    LOGGER.info(
        "Benchmark complete: %s, %s, %s",
        csv_path,
        metadata_path,
        ", ".join(str(path) for path in figure_paths),
    )


if __name__ == "__main__":
    main()
