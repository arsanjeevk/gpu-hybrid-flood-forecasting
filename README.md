# GPU Acceleration of Hybrid Physics-AI Flood Models for Real-Time Urban Flood Forecasting

This research project develops a real-time urban flood forecasting workflow
that combines an ANUGA reference model, a GPU-accelerated JAX finite-volume
shallow-water solver, and a learned residual correction network. The initial
study focuses on reproducible terrain and forcing preparation, numerical
validation, hybrid forecast accuracy, and matched JAX-versus-PyTorch
performance evaluation.

## Repository structure

- `config/` contains Hydra-style component and experiment configuration.
- `data/raw/` contains original read-only GIS and rainfall inputs.
- `data/interim/`, `data/synthetic_dem/`, and `data/processed/` contain
  reproducible derived datasets.
- `data/outputs/` contains solver runs and generated figures.
- `src/hybrid_flood/` contains the reusable Python package.
- `scripts/` contains thin command-line pipeline entry points.
- `notebooks/` is reserved for exploratory analysis.
- `tests/` contains numerical and model validation tests.
- `report/` contains the LaTeX research report.
- `docs/` contains methodology notes and architectural decisions.

## Setup

Install Python 3.11 and
[uv](https://docs.astral.sh/uv/), then run:

```bash
cd ~/Projects/hybrid-flood-model
uv sync
source .venv/bin/activate
```

The CUDA 12 JAX build is configured by default. Install the optional PyTorch
benchmark dependencies only when working on the comparison phase:

```bash
uv sync --extra pytorch-bench
```

## JAX execution devices

The CUDA build requires an NVIDIA GPU and a working host NVIDIA driver; CUDA
cannot use an AMD GPU. On a CPU-only workstation, the JAX solver detects the
missing driver, runs cleanly on CPU, and marks the GPU benchmark unavailable.
To inspect devices without asking JAX to initialize the installed CUDA plugin,
run:

```bash
JAX_PLATFORMS=cpu uv run python -c "import jax; print(jax.devices())"
```

For report-quality GPU measurements on an NVIDIA CUDA node, reject accidental
CPU fallback:

```bash
nvidia-smi
uv run python -c "import jax; print(jax.devices())"
uv run python scripts/04_run_jax_solver.py jax_solver.execution.require_gpu=true
uv run python scripts/08_benchmark_jax_vs_pytorch.py
```

The device list must include `CudaDevice(...)`. Benchmark JSON records the
device, default backend, timing, and whether a GPU benchmark was available.
The framework-comparison script additionally requires CUDA to be visible to
both JAX and PyTorch. It writes repeated timing results, a LaTeX table,
peak-memory metadata, profiler traces, and a PDF/PNG chart under
`data/outputs/benchmarks/`, `data/outputs/figures/`, and `report/figures/`.
It exits with an error rather than silently publishing CPU measurements when
the GPU requirement is not met.
