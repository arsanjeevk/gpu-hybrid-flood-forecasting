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
uv sync --frozen
source .venv/bin/activate
```

The CUDA 12 JAX build is configured by default. Install the optional PyTorch
benchmark dependencies only when working on the comparison phase:

```bash
uv sync --frozen --extra pytorch-bench
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
`data/outputs/figures/` and copies the CSV, LaTeX table, and chart into
`report/figures/`.
It exits with an error rather than silently publishing CPU measurements when
the GPU requirement is not met.

## Three-hour publication run

The publication configuration uses a 10,800 s development window with 181
one-minute outputs. It covers 8.82% of the available rainfall record,
integrates 8.35497 mm under the linearly interpolated forcing, and excludes the
record peak at hour 34. It must not be described as a complete return-period
event.

Stage the frozen Python/CUDA environment before billed GPU time when the login
and compute nodes share a filesystem:

```bash
uv sync --frozen --extra pytorch-bench
```

On the allocated A100 node, first run the fail-fast device, version,
configuration, input, and storage check:

```bash
uv run python scripts/00_gpu_preflight.py
```

Do not continue unless it writes a `passed` result to
`data/outputs/gpu_preflight.json`. Then execute the dependency-ordered
pipeline:

```bash
uv run python scripts/03_run_anuga_baseline.py
uv run python scripts/04_run_jax_solver.py
uv run python scripts/05_build_training_dataset.py
uv run python scripts/06_train_residual_net.py
uv run python scripts/07_run_hybrid_forecast.py
uv run python scripts/08_benchmark_jax_vs_pytorch.py
uv run python scripts/09_generate_report_figures.py
uv run python scripts/10_validate_gpu_results.py
uv run pytest tests -v
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

The final validator rejects CPU-generated or stale-duration artifacts,
non-finite fields, excessive mass/volume error, a hybrid model that fails to
improve held-out depth RMSE, incomplete benchmark repetitions, missing GPU
memory measurements, and missing report figures. Only after it passes should
the numerical values in `report/sections/` be updated from the new JSON/CSV
artifacts and `report/main.pdf` be rebuilt.

The hybrid correction relaxation is chosen from a fixed candidate set using
only the chronological validation block. Candidate rollouts begin at time zero
to include accumulated feedback error; the held-out test block is evaluated
only after selection.
