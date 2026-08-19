"""Fail-fast validation of the Google Colab NVIDIA T4 environment."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from hybrid_flood.jax_solver.runtime import configure_jax_runtime

JAX_RUNTIME = configure_jax_runtime()

import anuga  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from hybrid_flood.benchmark.jax_vs_pytorch import _normalized_gpu_name  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DURATION_S = 10_800.0


def _load(relative_path: str):
    return OmegaConf.load(ROOT / relative_path)


def _nvidia_smi() -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first = completed.stdout.strip().splitlines()[0]
    name, driver, memory_mib = (part.strip() for part in first.split(",", maxsplit=2))
    return {"name": name, "driver_version": driver, "memory_total_mib": memory_mib}


def main() -> None:
    """Validate one common T4, CUDA execution, inputs, and writable capacity."""
    errors: list[str] = []
    if sys.version_info[:2] != (3, 11):
        errors.append(f"Python 3.11 is required, found {sys.version.split()[0]}.")
    if str(anuga.__version__) != "3.3.7":
        errors.append(f"ANUGA 3.3.7 is required, found {anuga.__version__}.")
    if not str(torch.version.cuda or "").startswith("12."):
        errors.append(f"PyTorch must use CUDA 12.x, found {torch.version.cuda!r}.")

    jax_gpus = [device for device in jax.devices() if device.platform == "gpu"]
    if not jax_gpus:
        errors.append(f"JAX found no GPU: {jax.devices()}.")
    if not torch.cuda.is_available():
        errors.append("PyTorch reports torch.cuda.is_available() == False.")
    smi: dict[str, str] | None = None
    try:
        smi = _nvidia_smi()
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        errors.append(f"nvidia-smi probe failed: {exc}.")

    if jax_gpus and torch.cuda.is_available():
        jax_name = str(getattr(jax_gpus[0], "device_kind", jax_gpus[0]))
        torch_name = torch.cuda.get_device_name(0)
        if _normalized_gpu_name(jax_name) != _normalized_gpu_name(torch_name):
            errors.append(f"Framework GPU names differ: JAX={jax_name!r}, PyTorch={torch_name!r}.")
        required_name = str(_load("config/platform/colab_t4.yaml").required_gpu_name)
        if required_name.upper() not in torch_name.upper():
            errors.append(f"Expected NVIDIA {required_name}, found {torch_name!r}.")
        jax_value = jnp.ones((1024, 1024), dtype=jnp.float32)
        jax.block_until_ready(jax_value @ jax_value)
        torch_value = torch.ones((1024, 1024), device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        del torch_value

    anuga_cfg = _load("config/anuga/baseline.yaml")
    comparison_cfg = _load("config/comparison/v1_v2_t4.yaml")
    model_cfg = _load("config/model/residual_cnn.yaml")
    hybrid_cfg = _load("config/hybrid/coupled.yaml")
    viz_cfg = _load("config/viz/report.yaml")
    durations = (float(anuga_cfg.duration_s), float(comparison_cfg.common.duration_s))
    if durations != (EXPECTED_DURATION_S, EXPECTED_DURATION_S):
        errors.append(f"Both solver durations must be {EXPECTED_DURATION_S}, found {durations}.")
    if not bool(comparison_cfg.hardware.require_t4):
        errors.append("The active comparison does not require a T4.")
    if not bool(model_cfg.execution.require_gpu):
        errors.append("Training publication configuration does not require a GPU.")
    if not bool(hybrid_cfg.execution.require_gpu):
        errors.append("Hybrid publication configuration does not require a GPU.")
    if any(float(value) < 0 or float(value) > EXPECTED_DURATION_S for value in viz_cfg.key_times_s):
        errors.append("A configured report-figure time lies outside the simulation window.")
    if EXPECTED_DURATION_S % float(anuga_cfg.outputstep_s) != 0:
        errors.append("ANUGA output interval does not divide the duration exactly.")
    if EXPECTED_DURATION_S % float(comparison_cfg.common.output_interval_s) != 0:
        errors.append("Forecast output interval does not divide the duration exactly.")

    for configured in (
        *anuga_cfg.inputs.values(),
        comparison_cfg.common.inputs.dem,
        comparison_cfg.common.inputs.roughness,
        comparison_cfg.common.inputs.domain,
        comparison_cfg.common.inputs.rainfall,
        model_cfg.inputs.rainfall,
    ):
        path = ROOT / str(configured)
        if not path.is_file():
            errors.append(f"Required input is missing: {path}.")
    free_gib = shutil.disk_usage(ROOT).free / 1024**3
    minimum_free_gib = float(_load("config/platform/colab_t4.yaml").minimum_free_storage_gib)
    if free_gib < minimum_free_gib:
        errors.append(
            f"Only {free_gib:.2f} GiB is free; at least {minimum_free_gib} GiB is required."
        )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "failed" if errors else "passed",
        "errors": errors,
        "python": sys.version.split()[0],
        "anuga": str(anuga.__version__),
        "jax": str(jax.__version__),
        "jax_devices": [str(device) for device in jax.devices()],
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "nvidia_smi": smi,
        "jax_runtime_probe": JAX_RUNTIME,
        "simulation_duration_s": EXPECTED_DURATION_S,
        "free_storage_gib": free_gib,
    }
    output = ROOT / "data/outputs/t4_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("GPU preflight failed; do not start the publication pipeline.")


if __name__ == "__main__":
    main()
