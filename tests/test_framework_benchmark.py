"""Architecture-parity and benchmark-harness regression tests."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from hybrid_flood.benchmark.jax_vs_pytorch import (  # noqa: E402
    benchmark_frameworks,
    count_jax_parameters,
    format_latex_table,
)
from hybrid_flood.benchmark.profiling import summarize_chrome_trace  # noqa: E402
from hybrid_flood.benchmark.pytorch_residual_net import (  # noqa: E402
    TorchResidualUNet,
    count_torch_parameters,
)
from hybrid_flood.ml.residual_net import ResidualUNet  # noqa: E402
from hybrid_flood.viz.static_figures import plot_framework_benchmark  # noqa: E402


def _models() -> tuple[ResidualUNet, TorchResidualUNet]:
    flax_model = ResidualUNet(
        depth=3,
        channels=(16, 32, 64),
        activation="gelu",
        kernel_size=3,
        output_channels=3,
    )
    torch_model = TorchResidualUNet(
        input_channels=6,
        depth=3,
        channels=(16, 32, 64),
        activation="gelu",
        kernel_size=3,
        output_channels=3,
    )
    return flax_model, torch_model


def test_matched_models_have_identical_parameters_and_output_shapes() -> None:
    flax_model, torch_model = _models()
    jax_inputs = jnp.zeros((1, 129, 131, 6), dtype=jnp.float32)
    params = flax_model.init(jax.random.PRNGKey(0), jax_inputs)["params"]
    jax_output = flax_model.apply({"params": params}, jax_inputs)
    torch_output = torch_model(torch.zeros((1, 6, 129, 131)))
    assert count_jax_parameters(params) == count_torch_parameters(torch_model) == 150_883
    assert jax_output.shape == (1, 129, 131, 3)
    assert tuple(torch_output.shape) == (1, 3, 129, 131)


def test_cpu_smoke_benchmark_is_repeated_and_explicitly_labelled() -> None:
    flax_model, _ = _models()

    def factory() -> TorchResidualUNet:
        return _models()[1]

    results = benchmark_frameworks(
        flax_model,
        factory,
        batch_sizes=[1],
        input_shape=(16, 16, 6),
        warmup_iterations=0,
        measured_iterations=2,
        require_gpu=False,
        allow_cpu=True,
        torch_compile_enabled=False,
    )
    assert len(results) == 4
    assert set(results.framework) == {"JAX", "PyTorch"}
    assert set(results.operation) == {"forward", "forward_backward_adamw"}
    assert set(results.device_type) == {"cpu"}
    assert (results.measured_iterations == 2).all()
    assert np.isfinite(results[["mean_ms", "std_ms"]]).all().all()
    assert results.parameter_count.nunique() == 1


def test_gpu_requirement_rejects_cpu_only_execution() -> None:
    if any(device.platform == "gpu" for device in jax.devices()) and torch.cuda.is_available():
        pytest.skip("This assertion is specific to a CPU-only test host.")
    flax_model, _ = _models()
    with pytest.raises(RuntimeError, match="requires both JAX and PyTorch"):
        benchmark_frameworks(
            flax_model,
            lambda: _models()[1],
            batch_sizes=[1],
            input_shape=(16, 16, 6),
            warmup_iterations=0,
            measured_iterations=2,
            require_gpu=True,
        )


def test_latex_plot_and_trace_summary_exports(tmp_path) -> None:
    results = pd.DataFrame(
        [
            {
                "framework": framework,
                "operation": operation,
                "batch_size": batch,
                "mean_ms": float(batch),
                "std_ms": 0.1,
                "device_type": "gpu",
                "compile_or_first_iteration_ms": 5.0,
                "peak_memory_mb": 10.0,
            }
            for operation in ("forward", "forward_backward_adamw")
            for batch in (1, 4)
            for framework in ("JAX", "PyTorch")
        ]
    )
    table = format_latex_table(results, tmp_path / "benchmark.tex")
    figures = plot_framework_benchmark(results, tmp_path / "benchmark")
    assert table.is_file()
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)

    trace = {
        "traceEvents": [
            {"ph": "X", "name": "convolution", "cat": "device", "dur": 10.0},
            {"ph": "X", "name": "convolution", "cat": "device", "dur": 14.0},
        ]
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    summary_path = summarize_chrome_trace(
        trace_path,
        framework="Test",
        output_path=tmp_path / "summary.csv",
    )
    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "calls"] == 2
    assert summary.loc[0, "total_duration_us"] == 24.0
