"""Validate final three-hour GPU artifacts before they are cited in the report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DURATION_S = 10_800.0
EXPECTED_TIMES = 181


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Reject stale, CPU, incomplete, non-finite, or scientifically failed outputs."""
    errors: list[str] = []
    output = ROOT / "data/outputs/final_gpu_validation.json"
    required = (
        ROOT / "data/outputs/gpu_preflight.json",
        ROOT / "data/processed/residual_training_dataset.json",
        ROOT / "data/outputs/jax_runs/residual_net_run.json",
        ROOT / "data/processed/comparison_rollout.metadata.json",
        ROOT / "data/processed/anuga_baseline.nc",
        ROOT / "data/processed/jax_solver_raw.nc",
        ROOT / "data/processed/comparison_rollout.nc",
        ROOT / "data/outputs/figures/jax_vs_pytorch.csv",
        ROOT / "data/outputs/figures/figure_manifest.json",
    )
    for path in required:
        if not path.is_file():
            errors.append(f"Required publication artifact is missing: {path}.")
    anuga_candidates = sorted(
        ROOT.glob("data/outputs/anuga_runs/run_*_metadata.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    jax_candidates = sorted(
        ROOT.glob("data/outputs/jax_runs/run_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not anuga_candidates:
        errors.append("No ANUGA run metadata artifact was found.")
    if not jax_candidates:
        errors.append("No JAX solver run metadata artifact was found.")
    if errors:
        report = {
            "status": "failed",
            "errors": errors,
            "simulation_duration_s": EXPECTED_DURATION_S,
            "expected_time_count": EXPECTED_TIMES,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("Final GPU validation failed; do not cite these artifacts.")

    preflight = _read(ROOT / "data/outputs/gpu_preflight.json")
    if preflight.get("status") != "passed":
        errors.append("GPU preflight did not pass on the publication compute node.")

    anuga_metadata = _read(anuga_candidates[-1])
    if float(anuga_metadata["configuration"]["duration_s"]) != EXPECTED_DURATION_S:
        errors.append("Latest ANUGA artifact is not the three-hour run.")
    if anuga_metadata["boundaries"]["policy"] != "all_reflective_final":
        errors.append("ANUGA boundary policy is not the finalized reflective policy.")
    if float(anuga_metadata["simulation"]["maximum_relative_mass_error"]) > 1.0e-6:
        errors.append("ANUGA relative mass-balance error exceeds 1e-6.")

    jax_metadata = _read(jax_candidates[-1])
    if float(jax_metadata["configuration"]["duration_s"]) != EXPECTED_DURATION_S:
        errors.append("Latest JAX artifact is not the three-hour run.")
    if not bool(jax_metadata["execution"]["gpu_accelerated"]):
        errors.append("Latest JAX solver artifact was not GPU-accelerated.")

    dataset_metadata = _read(ROOT / "data/processed/residual_training_dataset.json")
    if dataset_metadata["split_counts"] != {"train": 126, "validation": 27, "test": 27}:
        errors.append(f"Unexpected chronological split: {dataset_metadata['split_counts']}.")

    training_metadata = _read(ROOT / "data/outputs/jax_runs/residual_net_run.json")
    if not bool(training_metadata["execution"]["gpu_accelerated"]):
        errors.append("Residual-network artifact was not trained on GPU.")
    metrics = training_metadata["test_metrics"]
    for name in ("depth_rmse", "depth_mae", "critical_success_index"):
        if not np.isfinite(float(metrics[name])):
            errors.append(f"Training test metric {name} is non-finite.")

    hybrid_metadata = _read(ROOT / "data/processed/comparison_rollout.metadata.json")
    if not bool(hybrid_metadata["execution"]["gpu_accelerated"]):
        errors.append("Hybrid rollout artifact was not produced on GPU.")
    selection = hybrid_metadata.get("relaxation_selection", {})
    if (
        not bool(selection.get("enabled"))
        or selection.get("selection_partition") != "validation"
        or bool(selection.get("test_partition_used_for_selection", True))
        or selection.get("metric") != "validation_hybrid_depth_rmse"
    ):
        errors.append("Hybrid relaxation was not selected exclusively on validation depth RMSE.")
    candidates = selection.get("candidate_validation_depth_rmse_m", {})
    selected_relaxation = str(selection.get("selected_relaxation"))
    if len(candidates) != 5 or selected_relaxation not in candidates:
        errors.append("Hybrid relaxation selection record is incomplete.")
    elif not np.isclose(
        float(candidates[selected_relaxation]),
        min(float(value) for value in candidates.values()),
    ):
        errors.append("Selected hybrid relaxation is not the best validation candidate.")
    hybrid_metrics = hybrid_metadata["metrics"]
    if float(hybrid_metrics["test_hybrid_depth_rmse_skill_percent"]) <= 0:
        errors.append("Hybrid depth RMSE did not improve on the held-out chronological test block.")
    if float(hybrid_metrics["test_hybrid_minus_jax_water_volume_max_relative"]) > 1.0e-4:
        errors.append("Hybrid correction changed JAX water volume by more than 1e-4 relative.")

    for relative in (
        "data/processed/anuga_baseline.nc",
        "data/processed/jax_solver_raw.nc",
        "data/processed/comparison_rollout.nc",
    ):
        with xr.open_dataset(ROOT / relative) as dataset:
            times = np.asarray(dataset.time.values, dtype=np.float64)
            if dataset.sizes["time"] != EXPECTED_TIMES:
                errors.append(f"{relative} has {dataset.sizes['time']} rather than 181 times.")
            if (
                times.size < 2
                or not np.all(np.diff(times) > 0.0)
                or not np.allclose(np.diff(times), 60.0, rtol=0.0, atol=1.0e-6)
            ):
                errors.append(f"{relative} does not have a strict 60 s output cadence.")
            if not np.isclose(float(times[-1]), EXPECTED_DURATION_S):
                errors.append(f"{relative} does not end at 10800 s.")
            for name in ("depth", "x_velocity", "y_velocity"):
                if name not in dataset:
                    continue
                values = np.asarray(dataset[name].values)
                if np.isinf(values).any():
                    errors.append(f"{relative}:{name} contains infinite values.")
                finite_counts = np.isfinite(values).reshape((*values.shape[:-2], -1)).sum(axis=-1)
                if np.any(finite_counts == 0):
                    errors.append(f"{relative}:{name} has an output slice with no finite cells.")
                if name == "depth" and np.nanmin(values) < -1.0e-7:
                    errors.append(f"{relative}:depth contains materially negative values.")

    benchmark = pd.read_csv(ROOT / "data/outputs/figures/jax_vs_pytorch.csv")
    if len(benchmark) != 16:
        errors.append(f"Benchmark contains {len(benchmark)} rows rather than 16.")
    if set(benchmark["device_type"]) != {"gpu"}:
        errors.append(f"Benchmark contains non-GPU rows: {set(benchmark['device_type'])}.")
    if benchmark["parameter_count"].nunique() != 1:
        errors.append("JAX and PyTorch benchmark parameter counts differ.")
    numeric = benchmark[
        ["mean_ms", "std_ms", "compile_or_first_iteration_ms", "peak_memory_mb"]
    ].to_numpy()
    if not np.isfinite(numeric).all() or np.any(numeric < 0):
        errors.append("Benchmark timing or memory fields are invalid.")
    if (benchmark["measured_iterations"] < 50).any():
        errors.append("Benchmark contains fewer than 50 measured iterations.")

    manifest = _read(ROOT / "data/outputs/figures/figure_manifest.json")
    if float(manifest["simulation_duration_s"]) != EXPECTED_DURATION_S:
        errors.append("Figure manifest is not for the three-hour simulation.")
    for artifact in manifest["artifacts"]:
        path = Path(artifact)
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty figure artifact: {path}.")
    for relative in (
        "report/figures/jax_vs_pytorch.csv",
        "report/figures/jax_vs_pytorch.tex",
        "report/figures/jax_vs_pytorch_benchmark.pdf",
        "report/figures/jax_vs_pytorch_benchmark.png",
    ):
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing report benchmark artifact: {path}.")

    report = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "simulation_duration_s": EXPECTED_DURATION_S,
        "expected_time_count": EXPECTED_TIMES,
        "anuga_maximum_relative_mass_error": anuga_metadata["simulation"][
            "maximum_relative_mass_error"
        ],
        "hybrid_test_depth_skill_percent": hybrid_metrics["test_hybrid_depth_rmse_skill_percent"],
        "benchmark_rows": len(benchmark),
        "figure_artifact_count": len(manifest["artifacts"]),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("Final GPU validation failed; do not cite these artifacts.")


if __name__ == "__main__":
    main()
